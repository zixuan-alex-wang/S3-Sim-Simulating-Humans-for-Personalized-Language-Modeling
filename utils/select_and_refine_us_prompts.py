"""Select and refine prompts for each US persona profile.
Usage:
    uv run python scripts/select_and_refine_prompts.py
    uv run python scripts/select_and_refine_prompts.py --force
    uv run python scripts/select_and_refine_prompts.py --sample-size 120
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path
import yaml as _yaml
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
UTILS_PROMPTS = PROJECT_ROOT / "utils" / "prompts"
DEFAULT_PROFILES_DIR = PROJECT_ROOT / "data" / "refined_profiles" / "US"
DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "data" / "initial_prompts" / "prompts_mixed_taged.jsonl"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
with open(UTILS_PROMPTS / "utils_filter_batch.yaml") as f:
    _FILTER_PROMPT: str = _yaml.safe_load(f)["prompt"]
with open(UTILS_PROMPTS / "utils_prompt_refiner.yaml") as f:
    _REFINER_PROMPT: str = _yaml.safe_load(f)["prompt"]

def render(template: str, **kw: str) -> str:
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template

def format_prompts_list(prompts: list[dict]) -> str:
    return "\n".join(f"[{i}] {p['text']}" for i, p in enumerate(prompts))

def load_profile(path: Path) -> dict:
    return _yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def save_profile(path: Path, data: dict) -> None:
    path.write_text(
        _yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )

def get_persona_summary(profile: dict) -> str:
    return profile.get("refined_summary") or profile.get("summary", "")

async def filter_prompts(
    llm,
    persona_summary: str,
    sampled: list[dict],
) -> list[dict]:
    prompts_list = format_prompts_list(sampled)
    rendered = render(_FILTER_PROMPT, persona_summary=persona_summary, prompts_list=prompts_list)
    result = await llm.chat_json(
        [
            {"role": "system", "content": rendered},
            {"role": "user", "content": "Evaluate and select prompts."},
        ],
        temperature=0.7,
        max_tokens=4096,
    )
    selected_indices = []
    for entry in result.get("selected", []):
        idx = entry.get("index")
        if idx is not None and 0 <= idx < len(sampled):
            selected_indices.append({
                "index": idx,
                "cluster": entry.get("cluster", ""),
                "score": entry.get("score", 0.0),
            })
    return selected_indices

async def refine_prompts(
    llm,
    persona_summary: str,
    sampled: list[dict],
    selected_indices: list[dict],
) -> list[dict]:
    refine_items = []
    for sel in selected_indices:
        idx = sel["index"]
        refine_items.append({"index": idx, "text": sampled[idx]["text"]})
    prompts_list = "\n".join(f"[{item['index']}] {item['text']}" for item in refine_items)
    rendered = render(_REFINER_PROMPT, persona_summary=persona_summary, selected_prompts_list=prompts_list)
    result = await llm.chat_json(
        [
            {"role": "system", "content": rendered},
            {"role": "user", "content": "Rewrite the selected prompts."},
        ],
        temperature=0.7,
        max_tokens=4096,
    )
    rewritten_map: dict[int, dict] = {}
    for entry in result.get("results", []):
        idx = entry.get("index")
        if idx is not None:
            rewritten_map[idx] = entry
    merged = []
    for sel in selected_indices:
        idx = sel["index"]
        prompt_data = sampled[idx]
        rw = rewritten_map.get(idx, {})
        merged.append({
            "prompt_id": prompt_data["id"],
            "original": prompt_data["text"],
            "rewritten": rw.get("rewritten", prompt_data["text"]),
            "cluster": sel.get("cluster", ""),
            "score": sel.get("score", 0.0),
        })
    return merged

async def process_profile(
    llm,
    profile_path: Path,
    all_prompts: list[dict],
    used_ids: set[str],
    sample_size: int,
    force: bool,
    idx: int,
    total: int,
) -> int:
    profile = load_profile(profile_path)
    profile_id = profile.get("persona_id", profile_path.stem)
    if not force and profile.get("selected_prompts"):
        log.info("[%d/%d] %s: already has selected_prompts, skipping", idx, total, profile_id)
        return 0
    available = [p for p in all_prompts if p["id"] not in used_ids]
    if len(available) < sample_size:
        log.warning(
            "[%d/%d] %s: only %d prompts available (need %d), using all remaining",
            idx, total, profile_id, len(available), sample_size,
        )
        sampled = available
    else:
        sampled = random.sample(available, sample_size)
    sampled_ids = {p["id"] for p in sampled}
    used_ids.update(sampled_ids)
    persona_summary = get_persona_summary(profile)
    log.info("[%d/%d] %s: sampling %d → filtering", idx, total, profile_id, len(sampled))
    try:
        selected_indices = await filter_prompts(llm, persona_summary, sampled)
    except Exception as e:
        log.error("[%d/%d] %s: filter failed: %s", idx, total, profile_id, e)
        return len(sampled)

    n_selected = len(selected_indices)
    log.info("[%d/%d] %s: %d selected → refining", idx, total, profile_id, n_selected)

    if n_selected == 0:
        log.warning("[%d/%d] %s: no prompts selected by filter, skipping refine", idx, total, profile_id)
        profile["selected_prompts"] = []
        save_profile(profile_path, profile)
        return len(sampled)
    try:
        merged = await refine_prompts(llm, persona_summary, sampled, selected_indices)
    except Exception as e:
        log.error("[%d/%d] %s: refine failed: %s", idx, total, profile_id, e)
        # Still save the filter results with originals as fallback
        merged = []
        for sel in selected_indices:
            sidx = sel["index"]
            merged.append({
                "prompt_id": sampled[sidx]["id"],
                "original": sampled[sidx]["text"],
                "rewritten": sampled[sidx]["text"],  # fallback: no rewrite
                "cluster": sel.get("cluster", ""),
                "score": sel.get("score", 0.0),
            })
    profile["selected_prompts"] = merged
    save_profile(profile_path, profile)

    log.info(
        "[%d/%d] %s: sampling %d → filtering → %d selected → refining → done",
        idx, total, profile_id, len(sampled), len(merged),
    )
    return len(sampled)

async def main():
    parser = argparse.ArgumentParser(description="Select and refine prompts for persona profiles")
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR,
                        help="Directory containing refined profile YAMLs")
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE,
                        help="Path to prompts JSONL file")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of prompts to sample per profile (default: 100)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process profiles that already have selected_prompts")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    random.seed(args.seed)
    if not args.profiles_dir.is_dir():
        log.error("Profiles directory not found: %s", args.profiles_dir)
        sys.exit(1)
    if not args.prompts_file.is_file():
        log.error("Prompts file not found: %s", args.prompts_file)
        sys.exit(1)
    all_prompts = []
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                all_prompts.append({
                    "id": d["prompt_id"],
                    "text": d["prompt_text"],
                    "fingerprint": d.get("fingerprint", {}),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    log.info("Loaded %d prompts from %s", len(all_prompts), args.prompts_file)
    profile_paths = sorted(args.profiles_dir.glob("profile_*.yaml"))
    if not profile_paths:
        log.error("No profile files found in %s", args.profiles_dir)
        sys.exit(1)
    log.info("Found %d profiles in %s", len(profile_paths), args.profiles_dir)
    used_ids: set[str] = set()
    if not args.force:
        for pp in profile_paths:
            data = load_profile(pp)
            existing = data.get("selected_prompts") or []
            for entry in existing:
                pid = entry.get("prompt_id")
                if pid:
                    used_ids.add(pid)
        if used_ids:
            log.info("Pre-populated used_ids with %d prompt IDs from existing profiles", len(used_ids))
    from utils.llm_client import LLMClient
    llm = LLMClient(max_concurrent=10)
    total = len(profile_paths)
    total_consumed = 0
    for i, pp in enumerate(profile_paths, 1):
        consumed = await process_profile(
            llm=llm,
            profile_path=pp,
            all_prompts=all_prompts,
            used_ids=used_ids,
            sample_size=args.sample_size,
            force=args.force,
            idx=i,
            total=total,
        )
        total_consumed += consumed
    log.info("Done. Processed %d profiles, consumed %d prompt slots. LLM stats: %s",
             total, total_consumed, llm.stats)

if __name__ == "__main__":
    asyncio.run(main())
