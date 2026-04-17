"""Extract profile fields from refined_profiles/US into two JSONL files.

Output:
  data/summary_refined_profiles_us.jsonl
    - persona_id, summary, refined_summary, behavioral_metadata
  data/original_rewritten_selected_prompts_us.jsonl
    - one line per prompt: {persona_id, prompt_id, original, rewritten}
"""
import json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = PROJECT_ROOT / "data" / "refined_profiles" / "US"
OUTPUT_DIR = PROJECT_ROOT / "data"

# Use ruamel or pyyaml
try:
    import yaml
except ImportError:
    sys.exit("pyyaml required: uv add pyyaml")


def main():
    profiles = sorted(PROFILES_DIR.glob("*.yaml"))
    print(f"Found {len(profiles)} profiles in {PROFILES_DIR}")

    summary_path = OUTPUT_DIR / "summary_refined_profiles_us.jsonl"
    prompts_path = OUTPUT_DIR / "original_rewritten_selected_prompts_us.jsonl"

    with open(summary_path, "w", encoding="utf-8") as f_summary, \
         open(prompts_path, "w", encoding="utf-8") as f_prompts:

        for p in profiles:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            pid = data.get("persona_id", p.stem)
            summary = data.get("summary", "")
            refined_summary = data.get("refined_summary", "")
            behavioral_metadata = data.get("behavioral_metadata", {})

            # File 1: summary + refined_summary + behavioral_metadata
            f_summary.write(json.dumps({
                "persona_id": pid,
                "summary": summary,
                "refined_summary": refined_summary,
                "behavioral_metadata": behavioral_metadata,
            }, ensure_ascii=False) + "\n")

            # File 2: one line per prompt
            for sp in data.get("selected_prompts", []):
                f_prompts.write(json.dumps({
                    "persona_id": pid,
                    "prompt_id": sp.get("prompt_id", ""),
                    "original": sp.get("original", ""),
                    "rewritten": sp.get("rewritten", ""),
                }, ensure_ascii=False) + "\n")

    print(f"Written {summary_path}")
    print(f"Written {prompts_path}")


if __name__ == "__main__":
    main()
