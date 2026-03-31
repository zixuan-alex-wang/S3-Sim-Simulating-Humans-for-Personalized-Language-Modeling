import json, logging, random, re
from pathlib import Path
import yaml
from user_simulator.data import Persona, Prompt, Scenario, LLM, fmt_conversation
from user_simulator.ablation import AblationConfig
from user_simulator.prompts import load_prompt, load_yaml, render
logger = logging.getLogger(__name__)
_ROOT_DIR = Path(__file__).resolve().parent.parent
_BEHAVIOR_TAXONOMY_PATH = _ROOT_DIR / "data" / "behavior_modes" / "tuna_behavior_taxonomy.yaml"
_BEHAVIOR_BLOCK_DIR = _ROOT_DIR / "data" / "behavior_modes" / "yaml"
def _load_prompt_any(*names: str) -> str:
    for name in names:
        try:
            return load_prompt(name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"None of prompt templates found: {names}")
_USER_TURN         = _load_prompt_any("user_s3")
_USER_TURN_VANILLA = _load_prompt_any("user_vanilla_with_behavior")
_USER_TURN_VANILLA_NO_BEHAVIOR = _load_prompt_any("user_vanilla")
_ASSISTANT         = _load_prompt_any("assistant_oracle", "assistant_vanilla", "assistant_vanilla_with_profile")
_SCENARIO          = load_prompt("simulator_lifelong_scenario_constructor")
try:
    _FILTER = _load_prompt_any("utils_filter_batch")
except FileNotFoundError:
    _FILTER = None
_BEHAVIOR_CONTROLLER_SPEC = load_yaml("simulator_behavior_sample")
_BEHAVIOR_CONTROLLER_USER = _BEHAVIOR_CONTROLLER_SPEC.get("user_prompt", "")

def _sync_behavior_blocks_from_taxonomy() -> dict:
    if not _BEHAVIOR_TAXONOMY_PATH.exists():
        return {}
    with open(_BEHAVIOR_TAXONOMY_PATH, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f) or {}
    behaviors = taxonomy.get("behaviors", {})
    if not isinstance(behaviors, dict):
        return taxonomy
    _BEHAVIOR_BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    for behavior_id, behavior in behaviors.items():
        if not isinstance(behavior, dict):
            continue
        if not (behavior.get("guidance_template") or "").strip():
            continue
        payload = dict(behavior)
        payload["behavior_id"] = behavior_id
        path = _BEHAVIOR_BLOCK_DIR / f"{behavior_id}.yaml"
        if path.exists():
            continue
        dumped = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        path.write_text(dumped, encoding="utf-8")
    return taxonomy

def _load_behavior_data() -> tuple[dict, dict, list[str], dict, dict]:
    taxonomy = _sync_behavior_blocks_from_taxonomy()
    projection = taxonomy.get("simulator_projection", {}) if isinstance(taxonomy, dict) else {}
    behaviors: dict[str, dict] = {}
    behavior_order: list[str] = []
    if _BEHAVIOR_BLOCK_DIR.exists():
        for path in sorted(_BEHAVIOR_BLOCK_DIR.glob("*.yaml")):
            if path.name == "example_behavior.yaml":
                continue
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                continue
            if not (raw.get("guidance_template") or "").strip():
                continue
            behavior_id = (raw.get("behavior_id") or path.stem).strip()
            raw["behavior_id"] = behavior_id
            behaviors[behavior_id] = raw
            behavior_order.append(behavior_id)
    if behaviors:
        behavior_order = _sort_behavior_ids_by_mode(behaviors, behavior_order)
        default_behavior = taxonomy.get("default_behavior", {})
        return taxonomy, behaviors, behavior_order, projection if isinstance(projection, dict) else {}, default_behavior
    if isinstance(taxonomy, dict) and isinstance(taxonomy.get("behaviors"), dict):
        source_behaviors = taxonomy.get("behaviors", {})
        for behavior_id, behavior in source_behaviors.items():
            if not isinstance(behavior, dict):
                continue
            if not (behavior.get("guidance_template") or "").strip():
                continue
            merged = dict(behavior)
            merged["behavior_id"] = behavior_id
            behaviors[behavior_id] = merged
            behavior_order.append(behavior_id)
        behavior_order = _sort_behavior_ids_by_mode(behaviors, behavior_order)
        default_behavior = source_behaviors.get("default_behavior", taxonomy.get("default_behavior", {}))
        return taxonomy, behaviors, behavior_order, projection if isinstance(projection, dict) else {}, default_behavior
    data = load_yaml("user_behaviors")
    source_behaviors = data.get("behaviors", {})
    for behavior_id, behavior in source_behaviors.items():
        if isinstance(behavior, dict) and (behavior.get("guidance_template") or "").strip():
            merged = dict(behavior)
            merged["behavior_id"] = behavior_id
            behaviors[behavior_id] = merged
            behavior_order.append(behavior_id)
    behavior_order = _sort_behavior_ids_by_mode(behaviors, behavior_order)
    return data, behaviors, behavior_order, {}, data.get("default_behavior", {})

def _sort_behavior_ids_by_mode(behaviors: dict[str, dict], behavior_order: list[str]) -> list[str]:
    mode_rank = {
        "Meta-Conversation": 0,
        "Social Interaction": 1,
        "Information Seeking": 2,
        "Information Processing & Synthesis": 3,
        "Procedural Guidance & Execution": 4,
        "Content Creation & Transformation": 5,
        "Multiple (blended)": 6,
        "Mixed": 7,
    }
    seen = set()
    ordered = []
    for behavior_id in behavior_order:
        if behavior_id in behaviors and behavior_id not in seen:
            ordered.append(behavior_id)
            seen.add(behavior_id)
    for behavior_id in behaviors:
        if behavior_id not in seen:
            ordered.append(behavior_id)
            seen.add(behavior_id)
    def sort_key(behavior_id: str):
        behavior = behaviors.get(behavior_id, {})
        mode = (behavior.get("tuna_mode") or "").strip()
        strategy = (behavior.get("tuna_strategy") or "").strip()
        name = (behavior.get("name") or behavior_id).strip()
        return (mode_rank.get(mode, 99), mode, strategy, name, behavior_id)
    return sorted(ordered, key=sort_key)

def _build_controller_behavior_catalog() -> str:
    rows = []
    mode_headers = {
        "Meta-Conversation": "Mode 6: Meta-Conversation",
        "Social Interaction": "Mode 5: Social Interaction",
        "Information Seeking": "Mode 1: Information Seeking",
        "Information Processing & Synthesis": "Mode 2: Information Processing",
        "Procedural Guidance & Execution": "Mode 3: Procedural Guidance & Execution",
        "Content Creation & Transformation": "Mode 4: Content Creation & Transformation",
        "Multiple (blended)": "Compound / Blended",
        "Mixed": "Default / Mixed",
    }
    current_mode = None
    for idx, behavior_id in enumerate(_BEHAVIOR_ORDER):
        behavior = _BEHAVIORS[behavior_id]
        mode = (behavior.get("tuna_mode") or "").strip()
        if mode != current_mode:
            current_mode = mode
            rows.append(f"## {mode_headers.get(mode, mode or 'Other')}")
        rows.append(f"[{idx}] id: {behavior_id}")
        rows.append(f"  name: {behavior.get('name', behavior_id)}")
        rows.append(f"  tuna_mode: {mode}")
        rows.append(f"  tuna_strategy: {behavior.get('tuna_strategy', '')}")
        rows.append(f"  cognitive_delegation_level: {behavior.get('cognitive_delegation_level', '')}")
        rows.append(f"  description: {(behavior.get('description', '') or '').strip().replace(chr(10), ' ')}")
    return "\n\n".join(rows) if rows else "[0] none | Natural Conversation"


_BEHAVIOR_DATA, _BEHAVIORS, _BEHAVIOR_ORDER, _SIMULATOR_PROJECTION, _DEFAULT_BEHAVIOR = _load_behavior_data()
_BEHAVIOR_CONTROLLER_SYSTEM = render(
    _BEHAVIOR_CONTROLLER_SPEC.get("system_prompt", ""),
    behavior_catalog=_build_controller_behavior_catalog(),
)

_INITIAL_USER_STATE = """**Conversation Progress:**
This is the start of the conversation. I just sent my opening message and am waiting for the assistant's response.

**My Current State:**
- Intent: {intent}
- Behavior Mode: exploring
- Predicted Next Action: follow_up

**Assessment of Assistant:**
No response received yet — this is the first turn.

**Next Action Plan:**
Waiting for the assistant's response. I'll evaluate how relevant and helpful it is, then decide what to ask next.
"""

def _strip_tags(text: str) -> str:
    return re.sub(r"</?(?:think|user_state|message|report|state)>", "", text).strip()

def _extract_json_object(text: str) -> dict:
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
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _extract_end_continue(msg: str) -> tuple[str, bool]:
    msg = msg.strip()
    for pattern, is_end in [("<|End Conversation|>", True), ("<|Continue Conversation|>", False)]:
        if msg.startswith(pattern):
            return msg[len(pattern):].strip(), is_end
    if "\n" in msg:
        first, rest = msg.split("\n", 1)
        first = first.strip()
        if first == "<|End Conversation|>":
            return rest.strip(), True
        if first == "<|Continue Conversation|>":
            return rest.strip(), False
    if msg.upper().startswith("END"):
        return msg[3:].strip().lstrip("\n"), True
    if msg.upper().startswith("CONTINUE"):
        return msg[8:].strip().lstrip("\n"), False
    return msg, False

def _parse_user_output(raw: str) -> dict:
    result = {"think": "", "user_state": "", "message": "", "wants_to_end": False}
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        result["think"] = m.group(1).strip()
    m = re.search(r"<user_state>(.*?)</user_state>", raw, re.DOTALL)
    if m:
        result["user_state"] = m.group(1).strip()
    else:
        m = re.search(r"\*\*Conversation Progress:\*\*(.+?)(?:\*\*Next Action Plan:\*\*.+?)$",
                       raw, re.DOTALL)
        if m:
            result["user_state"] = m.group(0).strip()
    msg = None
    m = re.search(r"<message>(.*?)</message>", raw, re.DOTALL)
    if m:
        msg = m.group(1).strip()
    if not msg:
        m = re.search(r"<message>(.*?)$", raw, re.DOTALL)
        if m:
            msg = m.group(1).strip()

    if not msg and "</user_state>" in raw:
        remainder = raw.split("</user_state>")[-1].strip()
        if remainder:
            msg = remainder
    if not msg:
        cleaned = raw
        for tag in ["think", "user_state"]:
            cleaned = re.sub(rf"<{tag}>.*?</{tag}>", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()
        if cleaned:
            msg = cleaned
    if not msg:
        msg = ""
    msg = _strip_tags(msg)
    msg, wants_to_end = _extract_end_continue(msg)
    result["message"] = msg
    result["wants_to_end"] = wants_to_end
    return result

def select_behavior(user_state: str | None = None) -> dict:
    names = [n for n, b in _BEHAVIORS.items() if isinstance(b, dict) and b.get("guidance_template")]
    if not names:
        return _DEFAULT_BEHAVIOR or {}
    default_weight = _SIMULATOR_PROJECTION.get("sampling", {}).get("default_weight", 1.0)
    weights = []
    for name in names:
        behavior = _BEHAVIORS[name]
        control = behavior.get("simulator_control", {})
        weights.append(control.get("sampling_weight", behavior.get("weight", default_weight)))
    chosen_name = random.choices(names, weights=weights, k=1)[0]
    chosen = dict(_BEHAVIORS[chosen_name])
    chosen["behavior_id"] = chosen_name
    return chosen

def _resolve_user_prompt_template(config: AblationConfig) -> str:
    if config.use_user_state:
        return "user_s3"
    if config.use_behavior_injection:
        return "user_vanilla_with_behavior"
    return "user_vanilla"

async def _select_behavior_with_controller(persona: Persona, conversation: list[dict],
                                           current_user_state: str, turn_number: int,
                                           total_turns: int, previous_behaviors: list[dict],
                                           llm: LLM) -> dict:
    fallback_behavior = select_behavior(current_user_state)
    if not _BEHAVIOR_CONTROLLER_SYSTEM or not _BEHAVIOR_CONTROLLER_USER:
        return {
            "behavior": fallback_behavior,
            "selected_behavior_index": None,
            "controller_source": "fallback",
        }
    previous_names = [item.get("behavior", "") for item in previous_behaviors if item.get("behavior")]
    previous_text = ", ".join(previous_names[-8:]) if previous_names else "N/A"
    user_prompt = render(
        _BEHAVIOR_CONTROLLER_USER,
        profile_summary=persona.summary[:1200] or "N/A",
        conversation_history=fmt_conversation(conversation[-12:]) or "N/A",
        turn_number=turn_number,
        total_turns=total_turns,
        previous_behaviors=previous_text,
        current_user_state=current_user_state or "N/A",
    )
    try:
        raw = await llm.chat(
            [{"role": "system", "content": _BEHAVIOR_CONTROLLER_SYSTEM},
             {"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=1200,
            call_type="behavior_controller",
        )
        decision = _extract_json_object(raw)
    except Exception as e:
        logger.warning("behavior controller failed at turn %d: %s", turn_number, e)
        return {
            "behavior": fallback_behavior,
            "selected_behavior_index": None,
            "controller_source": "fallback",
        }

    selected_idx_raw = decision.get("selected_behavior_index")
    selected_idx = None
    if isinstance(selected_idx_raw, int):
        selected_idx = selected_idx_raw
    elif isinstance(selected_idx_raw, str) and selected_idx_raw.isdigit():
        selected_idx = int(selected_idx_raw)
    behavior = fallback_behavior
    if isinstance(selected_idx, int) and 0 <= selected_idx < len(_BEHAVIOR_ORDER):
        selected_behavior_id = _BEHAVIOR_ORDER[selected_idx]
        behavior = dict(_BEHAVIORS[selected_behavior_id])
        behavior["behavior_id"] = selected_behavior_id
    include_few_shot = decision.get("include_few_shot")
    if behavior and isinstance(include_few_shot, bool):
        control = dict(behavior.get("simulator_control", {}))
        control["force_include_few_shot"] = include_few_shot
        behavior["simulator_control"] = control
    if behavior and decision.get("disclosure_stage") in {"minimal", "standard", "full"}:
        control = dict(behavior.get("simulator_control", {}))
        control["force_disclosure_stage"] = decision["disclosure_stage"]
        behavior["simulator_control"] = control
    return {
        "behavior": behavior,
        "selected_behavior_index": selected_idx,
        "controller_source": "llm",
    }

def _extract_request_types(behavior: dict) -> list[str]:
    types = []
    for item in behavior.get("few_shot_examples", []) or []:
        if isinstance(item, dict):
            req_type = item.get("request_type", "").strip()
            if req_type and req_type not in types:
                types.append(req_type)
    return types

def _extract_internal_question(template: str) -> str:
    m = re.search(r"\*\*Internal question:\*\*\s*(.+)", template)
    return m.group(1).strip() if m else ""

def _extract_section_bullets(template: str, title: str) -> list[str]:
    pattern = rf"\*\*{re.escape(title)}:\*\*(.*?)(?:\n\s*\*\*|$)"
    m = re.search(pattern, template, flags=re.DOTALL)
    if not m:
        return []
    lines = []
    for raw_line in m.group(1).splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            lines.append(line[2:].strip())
    return lines

def _infer_disclosure_stage(behavior: dict, conversation: list[dict]) -> str:
    forced = behavior.get("simulator_control", {}).get("force_disclosure_stage")
    if forced in {"minimal", "standard", "full"}:
        return forced
    assistant_turns = sum(1 for item in conversation if item.get("role") == "assistant")
    delegation = (behavior.get("cognitive_delegation_level") or "").lower()
    if "very high" in delegation or "high" in delegation:
        if assistant_turns <= 1:
            return "standard"
        return "full"
    if assistant_turns <= 1:
        return "minimal"
    if assistant_turns <= 4:
        return "standard"
    return "full"

def _make_behavior_block(behavior: dict | None, conversation: list[dict]) -> tuple[str, str, str]:
    if not behavior:
        return "", "none", "natural_flow"
    template = behavior.get("guidance_template", "")
    if not template.strip():
        return "", "none", "natural_flow"
    stage = _infer_disclosure_stage(behavior, conversation)
    behavior_id = behavior.get("behavior_id", "unknown")
    behavior_name = behavior.get("name", "Unnamed Behavior")
    request_types = _extract_request_types(behavior)
    rules = _extract_section_bullets(template, "Authenticity rules")
    selection_guidance = _extract_section_bullets(template, "Request type selection")
    if stage == "minimal":
        request_types = request_types[:3]
        rules = rules[:2]
        selection_guidance = selection_guidance[:2]
    elif stage == "standard":
        request_types = request_types[:5]
        rules = rules[:4]
        selection_guidance = selection_guidance[:4]
    internal_question = _extract_internal_question(template) if stage == "full" else ""
    examples = behavior.get("few_shot_examples", []) or []
    include_few_shot = behavior.get("simulator_control", {}).get("force_include_few_shot")
    if include_few_shot is False:
        examples = []
    if stage == "minimal":
        examples = []
    elif stage == "standard":
        examples = examples[:1]
    else:
        examples = examples[:2]
    examples_text = []
    for idx, item in enumerate(examples, start=1):
        if not isinstance(item, dict):
            continue
        user_turn = (item.get("user_turn", "") or "").strip().replace("\n", " ")
        request_type = item.get("request_type", "unknown")
        if user_turn:
            examples_text.append(f"{idx}. [{request_type}] {user_turn}")
    lines = [
        "<behavior_injection>",
        f"<behavior_id>{behavior_id}</behavior_id>",
        f"<behavior_name>{behavior_name}</behavior_name>",
        f"<behavior_mode>{behavior.get('tuna_mode', '')}</behavior_mode>",
        f"<behavior_strategy>{behavior.get('tuna_strategy', '')}</behavior_strategy>",
        f"<cognitive_delegation_level>{behavior.get('cognitive_delegation_level', '')}</cognitive_delegation_level>",
        f"<disclosure_stage>{stage}</disclosure_stage>",
        "<public_intent>",
        (behavior.get("description", "") or "").strip(),
        "</public_intent>",
    ]
    if request_types:
        lines.extend(["<public_request_types>", "\n".join(f"- {item}" for item in request_types), "</public_request_types>"])
    if selection_guidance:
        lines.extend(["<public_selection_guidance>", "\n".join(f"- {item}" for item in selection_guidance), "</public_selection_guidance>"])
    if rules:
        lines.extend(["<public_authenticity_rules>", "\n".join(f"- {item}" for item in rules), "</public_authenticity_rules>"])
    if examples_text:
        lines.extend(["<progressive_examples>", "\n".join(examples_text), "</progressive_examples>"])
    if internal_question:
        lines.extend(["<private_deliberation_focus>", internal_question, "</private_deliberation_focus>"])
    lines.append("</behavior_injection>")
    return "\n".join(lines), stage, behavior_name

async def generate_user_turn_vanilla(persona: Persona, conversation: list[dict],
                                     llm: LLM, behavior: dict | None = None,
                                     history_window: int | None = None,
                                     prompt_template_name: str | None = None) -> dict:
    """
    Generate user's next message WITHOUT state tracking (persona + history only).
    """
    behavior_block, behavior_stage, behavior_name = _make_behavior_block(behavior, conversation)

    # Build windowed conversation history
    if history_window is not None and history_window > 0:
        # Keep at most the last N messages (each user+assistant pair = 2 messages)
        window_msgs = conversation[-(history_window * 2):]
    else:
        window_msgs = conversation

    conversation_history = fmt_conversation(window_msgs)

    if prompt_template_name == "user_vanilla":
        prompt_template = _USER_TURN_VANILLA_NO_BEHAVIOR
    elif prompt_template_name == "user_vanilla_with_behavior":
        prompt_template = _USER_TURN_VANILLA
    else:
        prompt_template = _USER_TURN_VANILLA if behavior_block else _USER_TURN_VANILLA_NO_BEHAVIOR
    prompt = render(prompt_template,
                    persona_block=persona.to_block(),
                    behavior_block=behavior_block,
                    behavior_stage=behavior_stage,
                    behavior_name=behavior_name,
                    conversation_history=conversation_history)
    content = await llm.chat(
        [{"role": "system", "content": prompt},
         {"role": "user", "content": "Continue the conversation as this person."}],
        temperature=0.7, max_tokens=1024)
    msg = _strip_tags(content.strip())
    msg, wants_to_end = _extract_end_continue(msg)
    return {"message": msg, "wants_to_end": wants_to_end, "user_state": "", "think": ""}

async def generate_user_turn(persona: Persona, conversation: list[dict],
                             previous_user_state: str, llm: LLM,
                             behavior: dict | None = None) -> dict:
    """
    Generate user's next message with iterative state tracking.
    """
    behavior_block, behavior_stage, behavior_name = _make_behavior_block(behavior, conversation)
    prompt = render(_USER_TURN,
                    persona_block=persona.to_block(),
                    previous_user_state=previous_user_state,
                    behavior_block=behavior_block,
                    behavior_stage=behavior_stage,
                    behavior_name=behavior_name)
    messages = [{"role": "system", "content": prompt}] + conversation
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        content = await llm.chat(messages, temperature=0.7, max_tokens=2048)
        logger.debug("User sim content (attempt %d): %s", attempt, content[:300])
        result = _parse_user_output(content)
        if result["user_state"]:
            return result
        logger.warning("Empty user_state (attempt %d/%d), content starts: %s",
                       attempt, MAX_RETRIES, content[:200])
    # All retries failed — signal termination
    logger.error("All %d attempts returned empty user_state, terminating session", MAX_RETRIES)
    return {"message": "", "wants_to_end": True, "user_state": "", "think": "",
            "_terminated": "user_state_extraction_failed"}

async def rollout_conversation(persona: Persona, initial_prompt: str, prompt_id: str,
                               llm: LLM, max_turns=15, min_turns=5,
                               config: AblationConfig | None = None) -> dict:
    """
    Generate a full multi-turn conversation.
    """
    config = config or AblationConfig()
    conversation = [{"role": "user", "content": initial_prompt}]
    user_state_trajectory = []
    behavior_trajectory = []
    termination = "max_turns"
    current_user_state = _INITIAL_USER_STATE.format(
        intent=_guess_initial_intent(initial_prompt)
    ) if config.use_user_state else ""

    for turn in range(1, max_turns + 1):
        if config.assistant_strategy == "oracle":
            from user_simulator.prompts import load_prompt as _lp
            _oracle_tmpl = _lp("assistant_oracle")
            profile_summary = persona.refined_summary or persona.summary
            bm = persona.behavioral_metadata
            import json as _json
            bm_str = _json.dumps(bm, indent=2, ensure_ascii=False) if bm else "N/A"
            asst_prompt = render(_oracle_tmpl,
                                profile_summary=profile_summary,
                                behavior_metadata=bm_str,
                                conversation_prefix=fmt_conversation(conversation),
                                ground_truth_user_state=current_user_state or "N/A")
            asst_response = await llm.chat(
                [{"role": "system", "content": asst_prompt},
                 {"role": "user", "content": "Generate your response."}],
                temperature=0.7, max_tokens=1024)
        else:
            asst_msgs = [{"role": "system", "content": _ASSISTANT}] + conversation
            asst_response = await llm.chat(asst_msgs, temperature=0.7, max_tokens=1024)
        conversation.append({"role": "assistant", "content": asst_response})

        if turn >= max_turns:
            break
        selected_prompt_template = _resolve_user_prompt_template(config)
        if config.use_behavior_injection:
            controller_decision = await _select_behavior_with_controller(
                persona=persona,
                conversation=conversation,
                current_user_state=current_user_state,
                turn_number=turn,
                total_turns=max_turns,
                previous_behaviors=behavior_trajectory,
                llm=llm,
            )
            behavior = controller_decision.get("behavior")
        else:
            behavior = None
            controller_decision = {
                "controller_source": "disabled",
                "selected_behavior_index": None,
            }
        if behavior:
            behavior_trajectory.append({
                "turn": turn,
                "behavior": behavior.get("name", ""),
                "behavior_id": behavior.get("behavior_id", ""),
                "selected_behavior_index": controller_decision.get("selected_behavior_index"),
                "prompt_template": selected_prompt_template,
                "controller_source": controller_decision.get("controller_source", ""),
            })

        use_stateful_prompt = selected_prompt_template == "user_s3"
        if use_stateful_prompt:
            result = await generate_user_turn(
                persona, conversation, current_user_state, llm, behavior=behavior
            )
        else:
            behavior_for_turn = behavior if selected_prompt_template != "user_vanilla" else None
            result = await generate_user_turn_vanilla(
                persona, conversation, llm, behavior=behavior_for_turn,
                history_window=config.history_window,
                prompt_template_name=selected_prompt_template)
        if result.get("_terminated"):
            termination = result["_terminated"]
            logger.warning("Session terminated at turn %d: %s", turn, termination)
            break

        if use_stateful_prompt and result["user_state"]:
            current_user_state = result["user_state"]
        user_state_trajectory.append({
            "turn": turn,
            "think": result.get("think", ""),
            "user_state": current_user_state if use_stateful_prompt else "",
            "behavior": behavior.get("name", "") if behavior else "",
            "prompt_template": selected_prompt_template,
        })
        if result["message"]:
            conversation.append({"role": "user", "content": result["message"]})
        if result["wants_to_end"]:
            n_user = sum(1 for m in conversation if m["role"] == "user")
            if n_user >= min_turns:
                termination = "user_ended"
                break
            else:
                logger.debug("Override early end at turn %d (min=%d, current=%d)",
                             turn, min_turns, n_user)
        elif not result["message"]:
            logger.warning("Empty message at turn %d, ending rollout", turn)
            termination = "empty_message"
            break

    return {
        "persona_id": persona.id,
        "prompt_id": prompt_id,
        "conversation": conversation,
        "user_state_trajectory": user_state_trajectory,
        "behavior_trajectory": behavior_trajectory,
        "num_turns": sum(1 for m in conversation if m["role"] == "user"),
        "termination": termination,
        "ablation": config.name,
    }

def _guess_initial_intent(prompt: str) -> str:
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

async def generate_scenarios(persona: Persona, llm: LLM, n=7) -> list[Scenario]:
    prompt = render(_SCENARIO, profile_block=persona.to_block(), num_scenarios=n, persona_id=persona.id)
    raw = await llm.chat_json(
        [{"role": "system", "content": prompt},
         {"role": "user", "content": f"Generate {n} deeply personal scenarios."}],
        temperature=0.7, max_tokens=4096,
    )
    out = []
    for i, s in enumerate(raw.get("scenarios", [])):
        out.append(Scenario(
            id=s.get("scenario_id", f"{persona.id}_scenario_{i}"),
            category=s.get("category", "unknown"),
            initial_prompt=s.get("initial_prompt", ""),
            context_note=s.get("context_note", ""),
        ))
    return out

async def filter_prompts_batch(persona: Persona, prompts: list[Prompt],
                               llm: LLM, batch_size=20) -> list[Prompt]:
    suitable = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        prompts_text = "\n".join(f"[{j}] {p.text}" for j, p in enumerate(batch))
        prompt = render(_FILTER, persona_summary=persona.summary[:500], prompts_list=prompts_text)
        try:
            raw = await llm.chat_json(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": "Evaluate these prompts."}],
                temperature=0.7, max_tokens=2048,
            )
            for r in raw.get("results", []):
                idx = r.get("index", -1)
                if 0 <= idx < len(batch) and r.get("suitable", False):
                    suitable.append(batch[idx])
        except Exception as e:
            logger.warning("filter batch %d failed: %s", i, e)
    logger.info("Filtered %d → %d suitable prompts for %s", len(prompts), len(suitable), persona.id)
    return suitable