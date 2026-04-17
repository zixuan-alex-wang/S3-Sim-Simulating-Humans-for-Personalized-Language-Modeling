"""Data generation pipeline: rollout conversations with integrated assistant strategy.
No separate oracle annotation step — the assistant strategy (oracle/vanilla) is
applied directly during rollout. Results are streamed to disk as they complete.
Usage:
    uv run python -m user_simulator.pipeline --ablation full
    uv run python -m user_simulator.pipeline --ablation full --persona-ids profile_419 --max-prompts 2
    uv run python -m user_simulator.pipeline --ablation full --concurrency 80
"""
import argparse, asyncio, json, logging
from pathlib import Path
from user_simulator.data import (
    Persona, LLM, load_personas, save_json, CONV_DIR, SFT_DIR,
    SIM_MODEL, DATA_DIR,
)
from user_simulator.ablation import AblationConfig
from user_simulator.simulator import rollout_conversation
from user_simulator.oracle import build_sft_system_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR = DATA_DIR / "refined_profiles" / "US"


async def rollout_one(persona: Persona, prompt_data: dict, prompt_idx: int,
                      llm: LLM, config: AblationConfig,
                      conv_dir: Path, sft_file, sft_lock: asyncio.Lock,
                      max_turns: int, min_turns: int,
                      counter: dict):
    """Rollout a single conversation and stream-write the SFT instance."""
    prompt_id = prompt_data.get("prompt_id", f"prompt_{prompt_idx}")
    initial_msg = prompt_data.get("rewritten") or prompt_data.get("original", "")
    if not initial_msg:
        return

    safe_id = prompt_id.replace("/", "_").replace("\\", "_")
    conv_path = conv_dir / persona.id / f"{safe_id}.json"

    if conv_path.exists():
        counter["skipped"] += 1
        return

    try:
        session = await rollout_conversation(
            persona, initial_msg, prompt_id,
            llm, max_turns=max_turns, min_turns=min_turns,
            config=config,
        )
        session["profile_summary"] = persona.refined_summary
        session["behavioral_metadata"] = persona.behavioral_metadata
        session["scenario_category"] = prompt_data.get("cluster", "selected")

        # Save raw conversation JSON
        conv_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(session, conv_path)

        # Build SFT instance and stream-write
        sft_instance = _build_sft_instance(session, config)
        if sft_instance:
            async with sft_lock:
                sft_file.write(json.dumps(sft_instance, ensure_ascii=False) + "\n")
                sft_file.flush()

        counter["done"] += 1
        total = counter["total"]
        done = counter["done"]
        if done % 10 == 0 or done == total:
            logger.info("[%s] Progress: %d/%d done, %d skipped, %d failed",
                        config.name, done, total, counter["skipped"], counter["failed"])

    except Exception as e:
        counter["failed"] += 1
        logger.error("[%s/%s] Failed %s: %s", config.name, persona.id, safe_id, e)


def _build_sft_instance(session: dict, config: AblationConfig) -> dict | None:
    """Build a TRL multi-turn SFT instance from a completed session."""
    conversation = session.get("conversation", [])
    if not conversation:
        return None

    system_msg = build_sft_system_prompt(
        profile_summary=session.get("profile_summary", ""),
        behavior_metadata=json.dumps(session.get("behavioral_metadata", {}),
                                     indent=2, ensure_ascii=False)
            if session.get("behavioral_metadata") else "",
        include_profile=config.sft_include_profile,
    )

    messages = [{"role": "system", "content": system_msg}]
    for msg in conversation:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    return {
        "messages": messages,
        "metadata": {
            "persona_id": session.get("persona_id", ""),
            "scenario_id": session.get("prompt_id", ""),
            "num_turns": session.get("num_turns", 0),
            "termination": session.get("termination", ""),
            "ablation": config.name,
        },
    }


async def run(profiles_dir: Path = None, persona_ids: list[str] | None = None,
              max_turns: int = 12, min_turns: int = 3, max_prompts: int | None = None,
              concurrency: int = 80, do_assemble: bool = True,
              config: AblationConfig | None = None):
    """Run the pipeline with maximum concurrency."""
    config = config or AblationConfig()
    profiles_dir = profiles_dir or DEFAULT_PROFILES_DIR

    personas = load_personas(profiles_dir)
    if persona_ids:
        id_set = set(persona_ids)
        personas = [p for p in personas if p.id in id_set]
    personas = [p for p in personas if p.selected_prompts]

    # Build all (persona, prompt) tasks
    tasks_spec = []
    for persona in personas:
        prompts = persona.selected_prompts[:max_prompts] if max_prompts else persona.selected_prompts
        for idx, prompt_data in enumerate(prompts):
            tasks_spec.append((persona, prompt_data, idx))

    logger.info("Pipeline [%s]: %d personas, %d conversations, concurrency=%d, %d-%d turns",
                config.name, len(personas), len(tasks_spec), concurrency, min_turns, max_turns)

    llm = LLM(model=SIM_MODEL, max_concurrent=concurrency)
    conv_dir = CONV_DIR / config.name
    conv_dir.mkdir(parents=True, exist_ok=True)

    # Streaming SFT output
    sft_path = SFT_DIR / f"train_{config.name}.jsonl"
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    sft_lock = asyncio.Lock()

    counter = {"done": 0, "skipped": 0, "failed": 0, "total": len(tasks_spec)}

    # Open SFT file for streaming writes
    with open(sft_path, "w", encoding="utf-8") as sft_file:
        # Launch all tasks with semaphore-controlled concurrency
        sem = asyncio.Semaphore(concurrency)

        async def bounded(persona, prompt_data, idx):
            async with sem:
                await rollout_one(persona, prompt_data, idx,
                                  llm, config, conv_dir, sft_file, sft_lock,
                                  max_turns, min_turns, counter)

        await asyncio.gather(*[
            bounded(persona, prompt_data, idx)
            for persona, prompt_data, idx in tasks_spec
        ])

    logger.info("[%s] Complete: %d done, %d skipped, %d failed (of %d total)",
                config.name, counter["done"], counter["skipped"], counter["failed"],
                counter["total"])
    logger.info("[%s] SFT data → %s", config.name, sft_path)
    logger.info("LLM stats: %s", llm.stats)


def main():
    parser = argparse.ArgumentParser(description="S3-Sim data generation pipeline")
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument("--persona-ids", nargs="*", help="Specific persona IDs")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=80,
                        help="Max concurrent rollouts (default: 80)")
    parser.add_argument("--no-assemble", action="store_true")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_privilege", "no_behavior", "no_state"],
                        help="Experiment condition")
    args = parser.parse_args()

    config = AblationConfig.from_name(args.ablation)

    asyncio.run(run(
        profiles_dir=args.profiles_dir,
        persona_ids=args.persona_ids,
        max_turns=args.max_turns,
        min_turns=args.min_turns,
        max_prompts=args.max_prompts,
        concurrency=args.concurrency,
        config=config,
    ))


if __name__ == "__main__":
    main()
