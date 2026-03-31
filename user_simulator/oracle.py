"""
Oracle annotation and SFT data assembly. An oracle agent with privileged access to the user's profile and user_state generates personalized responses. For SFT, the system prompt includes the user profile so the student model learns to use it at inference time.
"""
import json, logging
from pathlib import Path
from user_simulator.data import Persona, LLM, count_tokens, fmt_conversation, load_json, ENC, CONV_DIR, SFT_DIR
from user_simulator.ablation import AblationConfig
from user_simulator.prompts import load_prompt, render
logger = logging.getLogger(__name__)
_ORACLE = load_prompt("assistant_oracle")
_ORACLE_NO_STATE = load_prompt("assistant_vanilla_with_profile")
BASE_SYSTEM_INSTRUCTION = (
    "You are a personalized AI assistant. Given the conversation so far, "
    "reason about the user's state and provide a helpful response."
)

def build_sft_system_prompt(profile_summary: str = "", behavior_metadata: str = "",
                            include_profile: bool = True) -> str:
    """Build the SFT system prompt, optionally injecting user profile."""
    parts = [BASE_SYSTEM_INSTRUCTION]
    if include_profile and profile_summary:
        parts.append(f"\n<user_profile>\n{profile_summary}\n</user_profile>")
    if include_profile and behavior_metadata:
        parts.append(f"\n<behavior_metadata>\n{behavior_metadata}\n</behavior_metadata>")
    return "\n".join(parts)

async def annotate_turn(persona: Persona, conversation: list[dict],
                        user_state: str, turn_idx: int, llm: LLM,
                        config: AblationConfig | None = None) -> dict:
    """
        Oracle generates a personalized response for one assistant turn.
    """
    config = config or AblationConfig()
    prefix = fmt_conversation(conversation)
    profile_summary = persona.metadata.get("refined_summary", "") or persona.summary
    behavior_metadata = json.dumps(persona.metadata.get("behavioral_metadata", {}),
                                   indent=2, ensure_ascii=False) if persona.metadata else "N/A"
    if config.oracle_has_user_state:
        prompt = render(_ORACLE,
                        profile_summary=profile_summary,
                        behavior_metadata=behavior_metadata,
                        conversation_prefix=prefix,
                        ground_truth_user_state=user_state)
    else:
        prompt = render(_ORACLE_NO_STATE,
                        profile_summary=profile_summary,
                        behavior_metadata=behavior_metadata,
                        conversation_prefix=prefix)

    content, thinking = await llm.chat(
        [{"role": "system", "content": prompt},
         {"role": "user", "content": "Generate your response."}],
        temperature=0.7, max_tokens=4096, return_thinking=True,
    )
    if thinking:
        oracle_output = f"<think>\n{thinking}\n</think>\n{content}"
    else:
        oracle_output = content

    return {
        "turn": turn_idx,
        "input": prefix,
        "output": oracle_output,
        "ground_truth_user_state": user_state,
        "total_tokens": count_tokens(prefix) + count_tokens(oracle_output),
    }

async def annotate_conversation(persona: Persona, session: dict, llm: LLM,
                                config: AblationConfig | None = None) -> list[dict]:
    """
    Annotate all assistant turns in a conversation.
    """
    config = config or AblationConfig()
    conversation = session["conversation"]
    us_traj = session.get("user_state_trajectory", [])
    instances = []
    us_by_turn = {s["turn"]: s["user_state"] for s in us_traj}
    for i, msg in enumerate(conversation):
        if msg["role"] != "assistant":
            continue
        prefix = conversation[:i]
        if not prefix:
            continue

        asst_idx = sum(1 for m in conversation[:i+1] if m["role"] == "assistant")
        user_state = us_by_turn.get(asst_idx, us_by_turn.get(asst_idx - 1, ""))

        if config.use_oracle:
            inst = await annotate_turn(persona, prefix, user_state, asst_idx, llm, config=config)
        else:
            inst = {
                "turn": asst_idx,
                "input": fmt_conversation(prefix),
                "output": msg["content"],
                "ground_truth_user_state": user_state,
                "total_tokens": count_tokens(fmt_conversation(prefix)) + count_tokens(msg["content"]),
            }
        inst["persona_id"] = persona.id
        inst["scenario_id"] = session.get("prompt_id", "")
        instances.append(inst)

    return instances

def assemble_sft(conversations_dir: Path = None, output_path: Path = None,
                 max_tokens: int = 32000):
    """Collect conversations into TRL-compatible multi-turn chat JSONL.

    Each line = one full multi-turn conversation:
        {"messages": [system, user, assistant, user, assistant, ...],
         "metadata": {...}}

    The system prompt includes the user profile for oracle experiments
    (ablation != "vanilla"), enabling the student to leverage profile info.
    """
    conversations_dir = Path(conversations_dir or CONV_DIR)
    output_path = Path(output_path or SFT_DIR / "train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    instances = []
    for persona_dir in sorted(conversations_dir.iterdir()):
        if not persona_dir.is_dir():
            continue
        for conv_file in sorted(persona_dir.rglob("*.json")):
            data = load_json(conv_file)
            conversation = data.get("conversation", [])
            oracle_anns = data.get("oracle_annotations", [])
            if not conversation:
                continue

            ablation_name = data.get("ablation", "full")
            include_profile = (ablation_name != "vanilla")
            system_msg = build_sft_system_prompt(
                profile_summary=data.get("profile_summary", ""),
                behavior_metadata=json.dumps(data.get("behavioral_metadata", {}),
                                             indent=2, ensure_ascii=False)
                    if data.get("behavioral_metadata") else "",
                include_profile=include_profile,
            )
            oracle_by_turn = {a["turn"]: a["output"] for a in oracle_anns if "output" in a}
            messages = [{"role": "system", "content": system_msg}]
            asst_idx = 0
            for msg in conversation:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    asst_idx += 1
                    oracle_out = oracle_by_turn.get(asst_idx)
                    content = oracle_out if oracle_out else msg["content"]
                    messages.append({"role": "assistant", "content": content})
            total_tokens = sum(count_tokens(m["content"]) for m in messages)
            if total_tokens > max_tokens:
                logger.debug("Skipping %s/%s: %d tokens > %d limit",
                             data.get("persona_id", "?"), data.get("prompt_id", "?"),
                             total_tokens, max_tokens)
                continue

            instances.append({
                "messages": messages,
                "metadata": {
                    "persona_id": data.get("persona_id", ""),
                    "scenario_id": data.get("prompt_id", ""),
                    "num_turns": data.get("num_turns", 0),
                    "termination": data.get("termination", ""),
                    "ablation": ablation_name,
                },
            })
    _write_jsonl(instances, output_path)
    logger.info("Assembled %d multi-turn SFT instances → %s", len(instances), output_path)
    return instances

def _write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")