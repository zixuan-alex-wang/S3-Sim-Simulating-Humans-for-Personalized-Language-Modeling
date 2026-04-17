"""Conversation quality analysis and state factor impact metrics.
Parses user_state_trajectory from generated conversations and computes:
- Behavior mode / intent / emotion distributions
- State transition matrices
- Conversation quality metrics (turns, termination, diversity)
"""
import json, logging, re
from collections import Counter, defaultdict
from pathlib import Path
from user_simulator.data import load_json, CONV_DIR

logger = logging.getLogger(__name__)


def _extract_field(user_state: str, field: str) -> str:
    """Extract a structured field from markdown user_state text."""
    m = re.search(rf"-\s*{field}:\s*(.+)", user_state, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip("[]\"'")
        return val.split("/")[0].split("(")[0].split("—")[0].strip().lower()
    return "unknown"


def parse_trajectory(us_traj: list[dict]) -> list[dict]:
    """Parse structured fields from a user_state_trajectory.

    Returns list of {turn, intent, emotion, behavior_mode, predicted_next_action, behavior_injected}
    """
    parsed = []
    for entry in us_traj:
        state = entry.get("user_state", "")
        parsed.append({
            "turn": entry.get("turn", 0),
            "intent": _extract_field(state, "Intent"),
            "emotion": _extract_field(state, "Emotion"),
            "behavior_mode": _extract_field(state, "Behavior Mode"),
            "predicted_next_action": _extract_field(state, "Predicted Next Action"),
            "behavior_injected": entry.get("behavior", ""),
        })
    return parsed


def compute_distributions(parsed_turns: list[dict]) -> dict:
    """Compute frequency distributions over state fields."""
    return {
        "intent": dict(Counter(t["intent"] for t in parsed_turns)),
        "behavior_mode": dict(Counter(t["behavior_mode"] for t in parsed_turns)),
        "predicted_next_action": dict(Counter(t["predicted_next_action"] for t in parsed_turns)),
        "behavior_injected": dict(Counter(t["behavior_injected"] for t in parsed_turns if t["behavior_injected"])),
    }


def compute_transition_matrix(parsed_turns: list[dict], field: str = "behavior_mode") -> dict:
    """Compute transition probabilities between states across consecutive turns.

    Returns: {from_state: {to_state: count}}
    """
    transitions = defaultdict(Counter)
    for i in range(len(parsed_turns) - 1):
        src = parsed_turns[i].get(field, "unknown")
        dst = parsed_turns[i + 1].get(field, "unknown")
        transitions[src][dst] += 1
    return {k: dict(v) for k, v in transitions.items()}


def analyze_conversations(conv_dir: Path = None, ablation: str = "full") -> dict:
    """Aggregate analysis across all conversations for an ablation condition.

    Args:
        conv_dir: root conversations directory (default: CONV_DIR)
        ablation: ablation name subdirectory

    Returns: {
        num_conversations, num_personas,
        turns: {mean, min, max, distribution},
        termination: {counts},
        distributions: {intent, behavior_mode, ...},
        transitions: {behavior_mode: {from: {to: count}}},
        diversity: {unique_intents, unique_behaviors, ...}
    }
    """
    conv_dir = Path(conv_dir or CONV_DIR) / ablation
    if not conv_dir.exists():
        logger.warning("No conversations found at %s", conv_dir)
        return {}

    all_parsed = []
    turn_counts = []
    terminations = Counter()
    num_personas = 0

    for persona_dir in sorted(conv_dir.iterdir()):
        if not persona_dir.is_dir():
            continue
        num_personas += 1
        for conv_file in sorted(persona_dir.glob("*.json")):
            data = load_json(conv_file)
            us_traj = data.get("user_state_trajectory", [])
            parsed = parse_trajectory(us_traj)
            all_parsed.extend(parsed)
            turn_counts.append(data.get("num_turns", 0))
            terminations[data.get("termination", "unknown")] += 1

    if not all_parsed:
        return {"num_conversations": 0}

    distributions = compute_distributions(all_parsed)
    transitions = compute_transition_matrix(all_parsed, "behavior_mode")

    return {
        "ablation": ablation,
        "num_conversations": len(turn_counts),
        "num_personas": num_personas,
        "num_turns_total": len(all_parsed),
        "turns": {
            "mean": sum(turn_counts) / len(turn_counts) if turn_counts else 0,
            "min": min(turn_counts) if turn_counts else 0,
            "max": max(turn_counts) if turn_counts else 0,
        },
        "termination": dict(terminations),
        "distributions": distributions,
        "transitions": transitions,
        "diversity": {
            "unique_intents": len(distributions["intent"]),
            "unique_behaviors": len(distributions["behavior_mode"]),
            "unique_actions": len(distributions["predicted_next_action"]),
        },
    }


def compare_ablations(ablation_names: list[str], conv_dir: Path = None) -> dict:
    """Run analysis for multiple ablation conditions and compare.

    Returns: {ablation_name: analysis_dict}
    """
    results = {}
    for name in ablation_names:
        results[name] = analyze_conversations(conv_dir, ablation=name)
    return results
