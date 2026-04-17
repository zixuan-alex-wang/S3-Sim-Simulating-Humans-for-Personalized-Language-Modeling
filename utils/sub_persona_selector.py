import json
import logging
from pathlib import Path
import yaml
from src.config import SUB_PERSONAS_DIR, DEFAULT_SUB_PERSONA_SLICES
from src.llm_client import LLMClient
from src.persona_loader import Persona
from src.prompts import load_prompt, render_prompt
logger = logging.getLogger(__name__)
_SELECTOR_TEMPLATE = load_prompt("sub_persona_selector")
def _format_profile_block(persona: Persona) -> str:
    """Format persona into XML-tagged profile block."""
    parts = [f"<persona_id>{persona.persona_id}</persona_id>"]
    if persona.attributes:
        parts.append(f"<attributes>\n{yaml.dump(persona.attributes, default_flow_style=False).strip()}\n</attributes>")
    else:
        parts.append("<attributes>empty</attributes>")
    parts.append(f"<summary>\n{persona.summary}\n</summary>")
    parts.append(f"<fingerprint>\n{json.dumps(persona.fingerprint, indent=2, ensure_ascii=False)}\n</fingerprint>")
    # Include behavioral metadata if available
    from src.profile_refiner import load_refined_profile
    refined = load_refined_profile(persona.persona_id)
    if refined and "behavioral_metadata" in refined:
        parts.append(f"<behavioral_metadata>\n{json.dumps(refined['behavioral_metadata'], indent=2, ensure_ascii=False)}\n</behavioral_metadata>")
    return "\n".join(parts)


async def generate_sub_personas(
    persona: Persona,
    client: LLMClient,
    num_slices: int = DEFAULT_SUB_PERSONA_SLICES,
) -> list[dict]:
    """Generate sub-persona slices for a persona."""
    profile_block = _format_profile_block(persona)
    system_msg = render_prompt(_SELECTOR_TEMPLATE,
        num_slices=num_slices,
        profile_block=profile_block,
    )
    user_msg = f"Generate exactly {num_slices} sub-persona slices for this persona."

    result = await client.chat_json(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=4096,
    )
    slices = result.get("slices", [])

    # Ensure slice IDs are properly set
    for i, s in enumerate(slices):
        if "slice_id" not in s:
            s["slice_id"] = f"{persona.persona_id}_slice_{i}"

    return slices


def save_sub_personas(persona_id: str, slices: list[dict], output_dir: Path = SUB_PERSONAS_DIR):
    """Save sub-persona slices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{persona_id}_slices.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"persona_id": persona_id, "slices": slices}, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d sub-persona slices for %s", len(slices), persona_id)


def load_sub_personas(persona_id: str, output_dir: Path = SUB_PERSONAS_DIR) -> list[dict]:
    """Load previously generated sub-persona slices."""
    path = output_dir / f"{persona_id}_slices.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("slices", [])


def is_sub_persona_done(persona_id: str, output_dir: Path = SUB_PERSONAS_DIR) -> bool:
    return (output_dir / f"{persona_id}_slices.json").exists()