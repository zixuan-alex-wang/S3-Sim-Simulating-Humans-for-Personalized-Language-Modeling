"""Conversation rollout engine for S3-Sim.

Supports two user simulator modes (stateful / vanilla) × two assistant
strategies (oracle / vanilla), controlled by AblationConfig.  Behavior
injection is orthogonal and driven by an LLM controller.
"""
import json, logging, random, re
from pathlib import Path
import yaml

from user_simulator.data import Persona, LLM, fmt_conversation
from user_simulator.ablation import AblationConfig
from user_simulator.prompts import load_prompt, load_yaml, render

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_BEHAVIOR_TAXONOMY_PATH = _ROOT / "data" / "behavior_modes" / "tuna_behavior_taxonomy.yaml"
_BEHAVIOR_BLOCK_DIR = _ROOT / "data" / "behavior_modes" / "yaml"

# ── Prompt templates (loaded once) ─────────────────────────────────────────────
_TMPL_USER_S3       = load_prompt("user_s3")
_TMPL_USER_VANILLA  = load_prompt("user_vanilla")
_TMPL_ASST_ORACLE   = load_prompt("assistant_oracle")
_TMPL_ASST_VANILLA  = load_prompt("assistant_vanilla")
_TMPL_ASST_ORACLE_PROFILE_ONLY = load_prompt("assistant_vanilla_with_profile")

_BEHAVIOR_SPEC      = load_yaml("simulator_behavior_sample")
_TMPL_CTRL_SYSTEM   = _BEHAVIOR_SPEC.get("system_prompt", "")
_TMPL_CTRL_USER     = _BEHAVIOR_SPEC.get("user_prompt", "")

# ── Persona field helpers ──────────────────────────────────────────────────────
def _persona_profile_summary(persona: Persona) -> str:
    return persona.refined_summary or persona.summary

def _persona_behavior_metadata_str(persona: Persona) -> str:
    bm = persona.behavioral_metadata
    return json.dumps(bm, indent=2, ensure_ascii=False) if bm else "N/A"


# ═══════════════════════════════════════════════════════════════════════════════
#  Behavior library
# ═══════════════════════════════════════════════════════════════════════════════

_MODE_RANK = {
    "Meta-Conversation": 0, "Social Interaction": 1,
    "Information Seeking": 2, "Information Processing & Synthesis": 3,
    "Procedural Guidance & Execution": 4, "Content Creation & Transformation": 5,
    "Multiple (blended)": 6, "Mixed": 7,
}

def _load_behaviors() -> tuple[dict[str, dict], list[str], dict, dict]:
    """Load behavior YAML blocks from data/behavior_modes/yaml/."""
    behaviors: dict[str, dict] = {}
    if _BEHAVIOR_BLOCK_DIR.exists():
        for p in sorted(_BEHAVIOR_BLOCK_DIR.glob("*.yaml")):
            if p.name.startswith("example") or p.suffix != ".yaml":
                continue
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict) or not (raw.get("guidance_template") or "").strip():
                continue
            bid = (raw.get("behavior_id") or p.stem).strip()
            raw["behavior_id"] = bid
            behaviors[bid] = raw

    def _key(bid: str):
        b = behaviors[bid]
        mode = (b.get("tuna_mode") or "").strip()
        return (_MODE_RANK.get(mode, 99), mode, b.get("tuna_strategy", ""), bid)
    order = sorted(behaviors, key=_key)

    taxonomy = {}
    if _BEHAVIOR_TAXONOMY_PATH.exists():
        taxonomy = yaml.safe_load(_BEHAVIOR_TAXONOMY_PATH.read_text(encoding="utf-8")) or {}
    projection = taxonomy.get("simulator_projection", {}) if isinstance(taxonomy, dict) else {}
    default = taxonomy.get("default_behavior", {}) if isinstance(taxonomy, dict) else {}
    return behaviors, order, projection, default

_BEHAVIORS, _BEHAVIOR_ORDER, _SIM_PROJECTION, _DEFAULT_BEHAVIOR = _load_behaviors()


def _build_behavior_catalog() -> str:
    """Indexed catalog string for the controller system prompt."""
    mode_labels = {
        "Meta-Conversation": "Mode 6: Meta-Conversation",
        "Social Interaction": "Mode 5: Social Interaction",
        "Information Seeking": "Mode 1: Information Seeking",
        "Information Processing & Synthesis": "Mode 2: Information Processing",
        "Procedural Guidance & Execution": "Mode 3: Procedural Guidance & Execution",
        "Content Creation & Transformation": "Mode 4: Content Creation & Transformation",
        "Multiple (blended)": "Compound / Blended",
        "Mixed": "Default / Mixed",
    }
    rows, cur_mode = [], None
    for idx, bid in enumerate(_BEHAVIOR_ORDER):
        b = _BEHAVIORS[bid]
        mode = (b.get("tuna_mode") or "").strip()
        if mode != cur_mode:
            cur_mode = mode
            rows.append(f"## {mode_labels.get(mode, mode or 'Other')}")
        desc = (b.get("description", "") or "").strip().replace("\n", " ")
        rows.append(
            f"[{idx}] id: {bid}\n"
            f"  name: {b.get('name', bid)}\n"
            f"  tuna_mode: {mode}\n"
            f"  tuna_strategy: {b.get('tuna_strategy', '')}\n"
            f"  cognitive_delegation_level: {b.get('cognitive_delegation_level', '')}\n"
            f"  description: {desc}"
        )
    return "\n\n".join(rows) if rows else "[0] none | Natural Conversation"

_CTRL_SYSTEM_RENDERED = render(_TMPL_CTRL_SYSTEM, behavior_catalog=_build_behavior_catalog())


# ═══════════════════════════════════════════════════════════════════════════════
#  Behavior selection
# ═══════════════════════════════════════════════════════════════════════════════

def _select_behavior_random() -> dict:
    """Weighted random selection from the behavior library."""
    names = [n for n in _BEHAVIOR_ORDER if _BEHAVIORS[n].get("guidance_template")]
    if not names:
        return _DEFAULT_BEHAVIOR or {}
    default_w = _SIM_PROJECTION.get("sampling", {}).get("default_weight", 1.0)
    weights = [_BEHAVIORS[n].get("weight", default_w) for n in names]
    chosen = random.choices(names, weights=weights, k=1)[0]
    out = dict(_BEHAVIORS[chosen])
    out["behavior_id"] = chosen
    return out


async def _select_behavior_with_controller(
    persona: Persona, conversation: list[dict], current_user_state: str,
    turn_number: int, total_turns: int, previous_behaviors: list[dict],
    llm: LLM,
) -> dict:
    """LLM controller picks the next behavior index."""
    prev_text = ", ".join(
        b.get("behavior", "") for b in previous_behaviors[-8:] if b.get("behavior")
    ) or "N/A"

    user_prompt = render(
        _TMPL_CTRL_USER,
        profile_summary=_persona_profile_summary(persona),
        behavior_metadata=_persona_behavior_metadata_str(persona),
        current_user_state=current_user_state or "N/A",
        conversation_prefix=fmt_conversation(conversation[-12:]) or "N/A",
        previous_behaviors=prev_text,
        turn_number=turn_number,
        total_turns=total_turns,
    )
    try:
        raw = await llm.chat(
            [{"role": "system", "content": _CTRL_SYSTEM_RENDERED},
             {"role": "user", "content": user_prompt}],
            temperature=0.9, max_tokens=128, json_mode=True,
            call_type="behavior_controller",
        )
        decision = _extract_json(raw)
    except Exception as e:
        logger.warning("Controller failed at turn %d: %s — raw: %s",
                       turn_number, e, raw[:200] if 'raw' in dir() else "N/A")
        return {"behavior": _select_behavior_random(), "controller_source": "fallback"}

    idx_raw = decision.get("selected_behavior_index")
    idx = int(idx_raw) if isinstance(idx_raw, (int, str)) and str(idx_raw).isdigit() else None

    if idx is not None and 0 <= idx < len(_BEHAVIOR_ORDER):
        behavior = dict(_BEHAVIORS[_BEHAVIOR_ORDER[idx]])
        behavior["behavior_id"] = _BEHAVIOR_ORDER[idx]
        logger.debug("Controller turn %d: idx=%d → %s", turn_number, idx, behavior["behavior_id"])
    else:
        behavior = _select_behavior_random()
        logger.warning("Controller turn %d: invalid idx=%s, raw decision=%s — using random: %s",
                       turn_number, idx_raw, decision, behavior.get("behavior_id"))

    # Apply controller overrides
    ctrl = dict(behavior.get("simulator_control", {}))
    if isinstance(decision.get("include_few_shot"), bool):
        ctrl["force_include_few_shot"] = decision["include_few_shot"]
    if decision.get("disclosure_stage") in {"minimal", "standard", "full"}:
        ctrl["force_disclosure_stage"] = decision["disclosure_stage"]
    if ctrl:
        behavior["simulator_control"] = ctrl

    return {"behavior": behavior, "controller_source": "llm"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Behavior block rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_disclosure_stage(behavior: dict, conversation: list[dict]) -> str:
    forced = behavior.get("simulator_control", {}).get("force_disclosure_stage")
    if forced in {"minimal", "standard", "full"}:
        return forced
    n_asst = sum(1 for m in conversation if m.get("role") == "assistant")
    delegation = (behavior.get("cognitive_delegation_level") or "").lower()
    high = "very high" in delegation or "high" in delegation
    if n_asst <= 1:
        return "standard" if high else "minimal"
    if n_asst <= 4:
        return "full" if high else "standard"
    return "full"


def _extract_bullets(template: str, title: str) -> list[str]:
    m = re.search(rf"\*\*{re.escape(title)}:\*\*(.*?)(?:\n\s*\*\*|$)", template, re.DOTALL)
    return [ln.strip()[2:].strip() for ln in (m.group(1) if m else "").splitlines()
            if ln.strip().startswith("- ")]


def _make_behavior_block(behavior: dict | None, conversation: list[dict]) -> tuple[str, str, str]:
    """Build <behavior_injection> XML block.  Returns (block, stage, name)."""
    if not behavior or not (behavior.get("guidance_template") or "").strip():
        return "", "none", "natural_flow"

    template = behavior["guidance_template"]
    stage = _infer_disclosure_stage(behavior, conversation)
    bid = behavior.get("behavior_id", "unknown")
    bname = behavior.get("name", bid)

    # Stage-based truncation
    clip = {"minimal": (3, 2), "standard": (5, 4), "full": (99, 99)}[stage]
    request_types = [
        it.get("request_type", "").strip()
        for it in (behavior.get("few_shot_examples") or [])
        if isinstance(it, dict) and it.get("request_type", "").strip()
    ][:clip[0]]
    rules = _extract_bullets(template, "Authenticity rules")[:clip[1]]
    guidance = _extract_bullets(template, "Request type selection")[:clip[1]]

    # Internal question (full stage only)
    iq_m = re.search(r"\*\*Internal question:\*\*\s*(.+)", template)
    internal_q = iq_m.group(1).strip() if iq_m and stage == "full" else ""

    # Few-shot examples — always included by default to ground behavior
    examples = behavior.get("few_shot_examples") or []
    force_fs = behavior.get("simulator_control", {}).get("force_include_few_shot")
    if force_fs is False:
        examples = []
    elif stage == "minimal":
        examples = examples[:2]
    elif stage == "standard":
        examples = examples[:3]
    else:
        examples = examples[:5]
    ex_lines = []
    for i, it in enumerate(examples, 1):
        if isinstance(it, dict) and it.get("user_turn"):
            ex_lines.append(f"{i}. [{it.get('request_type', '?')}] {it['user_turn'].strip()}")

    lines = [
        "<behavior_injection>",
        f"<behavior_id>{bid}</behavior_id>",
        f"<behavior_name>{bname}</behavior_name>",
        f"<behavior_mode>{behavior.get('tuna_mode', '')}</behavior_mode>",
        f"<behavior_strategy>{behavior.get('tuna_strategy', '')}</behavior_strategy>",
        f"<cognitive_delegation_level>{behavior.get('cognitive_delegation_level', '')}</cognitive_delegation_level>",
        f"<disclosure_stage>{stage}</disclosure_stage>",
        f"<public_intent>\n{(behavior.get('description') or '').strip()}\n</public_intent>",
    ]
    if request_types:
        lines.append(f"<public_request_types>\n{chr(10).join('- ' + t for t in request_types)}\n</public_request_types>")
    if guidance:
        lines.append(f"<public_selection_guidance>\n{chr(10).join('- ' + g for g in guidance)}\n</public_selection_guidance>")
    if rules:
        lines.append(f"<public_authenticity_rules>\n{chr(10).join('- ' + r for r in rules)}\n</public_authenticity_rules>")
    if ex_lines:
        lines.append(f"<progressive_examples>\n{chr(10).join(ex_lines)}\n</progressive_examples>")
    if internal_q:
        lines.append(f"<private_deliberation_focus>\n{internal_q}\n</private_deliberation_focus>")
    lines.append("</behavior_injection>")
    return "\n".join(lines), stage, bname


# ═══════════════════════════════════════════════════════════════════════════════
#  Output parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_tags(text: str) -> str:
    return re.sub(r"</?(?:think|user_state|message|report|state)>", "", text).strip()

def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        return json.loads(m.group(1))
    s, e = text.find("{"), text.rfind("}") + 1
    if s >= 0 and e > s:
        return json.loads(text[s:e])
    raise json.JSONDecodeError("No JSON object", text, 0)

def _extract_end_signal(msg: str) -> tuple[str, bool]:
    """Strip continuation/end tags, return (clean_msg, wants_to_end)."""
    msg = msg.strip()
    for tag, is_end in [("<|End Conversation|>", True), ("<|Continue Conversation|>", False)]:
        if msg.startswith(tag):
            return msg[len(tag):].strip(), is_end
    if "\n" in msg:
        first, rest = msg.split("\n", 1)
        first = first.strip()
        if first == "<|End Conversation|>":
            return rest.strip(), True
        if first == "<|Continue Conversation|>":
            return rest.strip(), False
    return msg, False


def _parse_user_output(raw: str) -> dict:
    """Parse <user_state> and <message> from user simulator output.

    Three extraction strategies for user_state, ordered by specificity:
      1. <user_state>...</user_state>  (exact tags)
      2. <user_state>...              (unclosed tag, stops at <message> or EOF)
      3. # User State Report...       (model omitted tags entirely)
    """
    result = {"think": "", "user_state": "", "message": "", "wants_to_end": False}

    # Think block
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        result["think"] = m.group(1).strip()

    # User state
    m = re.search(r"<user_state>(.*?)</user_state>", raw, re.DOTALL)
    if m:
        result["user_state"] = m.group(1).strip()
    else:
        m = re.search(r"<user_state>(.*?)(?=</?message>|$)", raw, re.DOTALL)
        if m and m.group(1).strip():
            result["user_state"] = m.group(1).strip()
        else:
            m = re.search(r"(#\s*User State Report.*?)(?=</?message>|\Z)", raw, re.DOTALL)
            if m and len(m.group(1).strip()) > 80:
                result["user_state"] = m.group(1).strip()

    # Message
    m = re.search(r"<message>(.*?)</message>", raw, re.DOTALL)
    if m:
        msg = m.group(1).strip()
    else:
        m = re.search(r"<message>(.*?)$", raw, re.DOTALL)
        if m and m.group(1).strip():
            msg = m.group(1).strip()
        elif "</user_state>" in raw:
            msg = raw.split("</user_state>")[-1].strip()
        else:
            msg = ""
    msg = _strip_tags(msg)
    msg, wants_to_end = _extract_end_signal(msg)
    result["message"] = msg
    result["wants_to_end"] = wants_to_end
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  User turn generators
# ═══════════════════════════════════════════════════════════════════════════════

_INITIAL_STATE = (
    "# User State Report\n\n"
    "## Explicit Conversarional Context\n\n"
    "Turn index: 1.\n"
    "Whether this is a new session or continuation: new session.\n\n"
    "### Cross turn memory\nN/A\n\n"
    "### Conversation log\n"
    "Just sent my opening message. Waiting for the assistant's first response.\n\n"
    "## Implicit User Inner State\n\n"
    "### Stable state\n"
    "1. Long-term goal: {intent}\n"
    "2. Beliefs: to be determined based on assistant quality\n"
    "3. Values: accuracy, relevance, respect for my expertise\n"
    "4. Background constraints: as described in profile\n"
    "5. Stance: neutrally observing\n\n"
    "### Dynamic state\n"
    "1. Behavior mode: exploring\n"
    "2. Short-term intent: {intent}\n"
    "3. Emotion: mild curiosity about how the assistant will respond\n"
    "4. Internal tension: none yet\n\n"
    "### Evaluation of Last assistant turn\n"
    "No response received yet.\n\n"
    "### Next action plan\n"
    "Waiting for the assistant's response. Will evaluate relevance and depth."
)


async def generate_user_turn(persona: Persona, conversation: list[dict],
                             previous_user_state: str, llm: LLM,
                             behavior: dict | None = None,
                             turn_number: int = 1, max_turns: int = 12) -> dict:
    """Stateful user turn: maintains <user_state> as structured working memory.

    Last 2 user+assistant rounds (4 messages) are passed as chat context.
    Full conversation history is encoded in previous_user_state (conversation log).
    This prevents the model from copying repetitive patterns in long histories
    while preserving enough recent context for coherent state updates.
    """
    behavior_block, stage, bname = _make_behavior_block(behavior, conversation)
    prompt = render(_TMPL_USER_S3,
                    profile_summary=_persona_profile_summary(persona),
                    behavior_metadata=_persona_behavior_metadata_str(persona),
                    previous_user_state=previous_user_state,
                    behavior_block=behavior_block,
                    behavior_stage=stage,
                    behavior_name=bname)

    # Keep last 2 rounds (4 messages) — enough for coherent state updates
    # without triggering repetition from long history.
    recent = conversation[-4:] if len(conversation) > 4 else conversation

    _FORMAT_REMINDER = (
        f"This is turn {turn_number + 1} of {max_turns} in an ONGOING session (not a new session). "
        "You MUST output exactly: <user_state>...</user_state> then <message>...</message>. "
        "Do NOT answer the question — you ARE the user. "
        "Your <user_state> MUST update Cross turn memory with what happened so far. "
        "Your new message MUST differ from previous messages and advance the conversation."
    )
    messages = [{"role": "system", "content": prompt}] + recent + [
        {"role": "user", "content": _FORMAT_REMINDER}
    ]

    # Escalate temperature on retry to break out of degenerate sampling modes
    RETRY_TEMPS = [0.7, 0.8, 0.9, 1.0, 1.1]
    for attempt, temp in enumerate(RETRY_TEMPS, 1):
        content = await llm.chat(messages, temperature=temp, max_tokens=2048)
        logger.debug("User sim (attempt %d, T=%.1f): %s", attempt, temp, content[:300])
        result = _parse_user_output(content)
        if result["user_state"]:
            return result
        logger.warning("Empty user_state (%d/%d T=%.1f), raw: %s",
                       attempt, len(RETRY_TEMPS), temp, content[:200])

    logger.error("All %d retries failed — terminating session", len(RETRY_TEMPS))
    return {"message": "", "wants_to_end": True, "user_state": "", "think": "",
            "_terminated": "user_state_extraction_failed"}


async def generate_user_turn_vanilla(persona: Persona, conversation: list[dict],
                                     llm: LLM, history_window: int | None = None) -> dict:
    """Vanilla user turn: persona + conversation history, no state tracking."""
    window = conversation[-(history_window * 2):] if history_window else conversation
    prompt = render(_TMPL_USER_VANILLA,
                    profile_summary=_persona_profile_summary(persona),
                    behavior_metadata=_persona_behavior_metadata_str(persona),
                    conversation_history=fmt_conversation(window))
    content = await llm.chat(
        [{"role": "system", "content": prompt},
         {"role": "user", "content": "Continue the conversation as this person."}],
        temperature=0.7, max_tokens=1024)
    msg = _strip_tags(content.strip())
    msg, wants_to_end = _extract_end_signal(msg)
    return {"message": msg, "wants_to_end": wants_to_end, "user_state": "", "think": ""}


# ═══════════════════════════════════════════════════════════════════════════════
#  Conversation rollout
# ═══════════════════════════════════════════════════════════════════════════════

def _guess_intent(prompt: str) -> str:
    p = prompt.lower()
    if "?" in p:
        if any(w in p for w in ["recommend", "suggest", "best", "should i"]):
            return "get_recommendation"
        if any(w in p for w in ["how to", "how do", "fix", "solve", "help me"]):
            return "solve_problem"
        return "seek_info"
    if any(w in p for w in ["feel", "stressed", "worried", "frustrated", "upset"]):
        return "vent"
    return "explore_topic"


async def rollout_conversation(
    persona: Persona, initial_prompt: str, prompt_id: str,
    llm: LLM, max_turns: int = 15, min_turns: int = 5,
    config: AblationConfig | None = None,
) -> dict:
    """Run a full multi-turn conversation rollout.

    User mode:  stateful (user_s3) or vanilla, per config.use_user_state.
    Assistant:  oracle (profile+state) or vanilla, per config.assistant_strategy.
    Behavior:   LLM-controller injection when config.use_behavior_injection=True.
    """
    config = config or AblationConfig()
    conversation = [{"role": "user", "content": initial_prompt}]
    us_trajectory, bh_trajectory = [], []
    termination = "max_turns"

    current_state = ""
    if config.use_user_state:
        current_state = _INITIAL_STATE.format(intent=_guess_intent(initial_prompt))

    profile_summary = _persona_profile_summary(persona)
    bm_str = _persona_behavior_metadata_str(persona)

    for turn in range(1, max_turns + 1):
        # ── Assistant turn ──
        if config.assistant_strategy == "oracle":
            asst_prompt = render(_TMPL_ASST_ORACLE,
                                profile_summary=profile_summary,
                                behavior_metadata=bm_str,
                                conversation_prefix=fmt_conversation(conversation),
                                ground_truth_user_state=current_state or "N/A")
            asst_response = await llm.chat(
                [{"role": "system", "content": asst_prompt},
                 {"role": "user", "content": "Generate your response."}],
                temperature=0.7, max_tokens=1024)
        elif config.assistant_strategy == "oracle_profile_only":
            asst_prompt = render(_TMPL_ASST_ORACLE_PROFILE_ONLY,
                                profile_summary=profile_summary,
                                behavior_metadata=bm_str,
                                conversation_prefix=fmt_conversation(conversation))
            asst_response = await llm.chat(
                [{"role": "system", "content": asst_prompt},
                 {"role": "user", "content": "Generate your response."}],
                temperature=0.7, max_tokens=1024)
        else:
            asst_prompt = render(_TMPL_ASST_VANILLA,
                                conversation_prefix=fmt_conversation(conversation))
            asst_response = await llm.chat(
                [{"role": "system", "content": asst_prompt},
                 {"role": "user", "content": "Generate your response."}],
                temperature=0.7, max_tokens=1024)
        conversation.append({"role": "assistant", "content": asst_response})

        if turn >= max_turns:
            break

        # ── Behavior selection ──
        behavior = None
        ctrl_src = "disabled"
        if config.use_behavior_injection:
            ctrl = await _select_behavior_with_controller(
                persona, conversation, current_state,
                turn, max_turns, bh_trajectory, llm,
            )
            behavior = ctrl["behavior"]
            ctrl_src = ctrl["controller_source"]

        # Build behavior block for this turn (used in user prompt + logged)
        behavior_block_text = ""
        if behavior:
            behavior_block_text, _, _ = _make_behavior_block(behavior, conversation)
            bh_trajectory.append({
                "turn": turn,
                "behavior": behavior.get("name", ""),
                "behavior_id": behavior.get("behavior_id", ""),
                "controller_source": ctrl_src,
                "guidance_block": behavior_block_text,
            })

        # ── User turn ──
        if config.use_user_state:
            result = await generate_user_turn(
                persona, conversation, current_state, llm, behavior=behavior,
                turn_number=turn, max_turns=max_turns)
        else:
            result = await generate_user_turn_vanilla(
                persona, conversation, llm, history_window=config.history_window)

        if result.get("_terminated"):
            termination = result["_terminated"]
            logger.warning("Terminated at turn %d: %s", turn, termination)
            break

        if config.use_user_state and result["user_state"]:
            current_state = result["user_state"]

        us_trajectory.append({
            "turn": turn,
            "think": result.get("think", ""),
            "user_state": current_state if config.use_user_state else "",
            "behavior": behavior.get("name", "") if behavior else "",
            "prompt_template": "user_s3" if config.use_user_state else "user_vanilla",
        })

        if result["message"]:
            conversation.append({"role": "user", "content": result["message"]})

        if result["wants_to_end"]:
            n_user = sum(1 for m in conversation if m["role"] == "user")
            if n_user >= min_turns:
                termination = "user_ended"
                break
        elif not result["message"]:
            termination = "empty_message"
            break

    return {
        "persona_id": persona.id,
        "prompt_id": prompt_id,
        "conversation": conversation,
        "user_state_trajectory": us_trajectory,
        "behavior_trajectory": bh_trajectory,
        "num_turns": sum(1 for m in conversation if m["role"] == "user"),
        "termination": termination,
        "ablation": config.name,
        "models": {
            "user_simulator": llm.model,
            "assistant": llm.model,
            "behavior_controller": llm.model,
        },
        "prompt_templates": {
            "user": "user_s3" if config.use_user_state else "user_vanilla",
            "assistant": {
                "oracle": "assistant_oracle",
                "oracle_profile_only": "assistant_vanilla_with_profile",
            }.get(config.assistant_strategy, "assistant_vanilla"),
        },
    }
