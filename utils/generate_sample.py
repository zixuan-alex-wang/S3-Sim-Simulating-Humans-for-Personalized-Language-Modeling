"""Generate sample multi-turn SFT data.
Pipeline:
1. Randomly pick 1 persona (or specify one)
2. Generate ~30 persona-grounded scenarios using the scenario generator
3. For each scenario: rollout state-based multi-turn conversation
4. Oracle-annotate assistant turns
5. Assemble into TRL multi-turn format (one full conversation per JSONL line)
Usage:
    uv run python generate_sample.py
    uv run python generate_sample.py --num-scenarios 10 --max-turns 8
    uv run python generate_sample.py --ablation vanilla --persona-id profile_102
"""
import argparse, asyncio, json, logging, random
from pathlib import Path

from user_simulator.data import (
    LLM, load_personas,
    save_json, CONV_DIR, SIM_MODEL, ORACLE_MODEL, DATA_DIR,
)
from user_simulator.ablation import AblationConfig
from user_simulator.simulator import rollout_conversation, generate_scenarios
from user_simulator.oracle import annotate_conversation, assemble_sft

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def generate_sample(
    num_scenarios: int = 30,
    max_turns: int = 12,
    min_turns: int = 3,
    persona_id: str | None = None,
    ablation: str = "full",
    output_dir: Path | None = None,
):
    config = AblationConfig.from_name(ablation)

    # 1. Load personas (from both profiles/yaml and filtered_profiles)
    logger.info("Loading personas...")
    personas = load_personas()
    # Also load from filtered_profiles subdirectories
    filtered_dir = DATA_DIR / "filtered_profiles"
    if filtered_dir.exists():
        for sub in filtered_dir.iterdir():
            if sub.is_dir():
                personas.extend(load_personas(sub))
    # Deduplicate by ID
    seen = set()
    unique = []
    for p in personas:
        if p.id not in seen:
            seen.add(p.id)
            unique.append(p)
    personas = unique
    logger.info("Total personas: %d", len(personas))

    # 2. Pick one persona
    if persona_id:
        persona = next((p for p in personas if p.id == persona_id), None)
        if not persona:
            raise ValueError(f"Persona {persona_id!r} not found")
    else:
        persona = random.choice(personas)
    logger.info("Selected persona: %s", persona.id)
    logger.info("Summary: %s", persona.summary[:300])

    # 3. Setup LLMs
    sim_llm = LLM(model=SIM_MODEL, max_concurrent=5)
    oracle_llm = LLM(model=ORACLE_MODEL, max_concurrent=5)

    # 4. Generate persona-grounded scenarios
    logger.info("Generating %d persona-grounded scenarios...", num_scenarios)
    scenarios = await generate_scenarios(persona, sim_llm, n=num_scenarios)
    logger.info("Generated %d scenarios: %s",
                len(scenarios), [s.category for s in scenarios])

    # 5. Output directory
    out_dir = output_dir or (CONV_DIR / f"sample_{config.name}")
    persona_dir = out_dir / persona.id
    persona_dir.mkdir(parents=True, exist_ok=True)

    # 6. Rollout + annotate each scenario
    results = []
    for i, scenario in enumerate(scenarios):
        logger.info("[%d/%d] %s | %s",
                    i + 1, len(scenarios), scenario.category,
                    scenario.initial_prompt[:80])
        try:
            session = await rollout_conversation(
                persona, scenario.initial_prompt, scenario.id,
                sim_llm, max_turns=max_turns, min_turns=min_turns,
                config=config,
            )
            session["scenario_category"] = scenario.category
            session["scenario_context_note"] = scenario.context_note

            logger.info("  %d turns (%s), annotating with oracle...",
                        session["num_turns"], session["termination"])

            oracle_instances = await annotate_conversation(
                persona, session, oracle_llm, config=config)
            session["oracle_annotations"] = oracle_instances

            # Save raw conversation
            safe_id = scenario.id.replace("/", "_").replace("\\", "_")
            conv_path = persona_dir / f"{safe_id}.json"
            save_json(session, conv_path)
            results.append(session)

            logger.info("  Done: %d oracle annotations", len(oracle_instances))

        except Exception as e:
            logger.error("  Failed: %s", e)

    logger.info("Completed %d/%d conversations", len(results), len(scenarios))
    logger.info("Sim LLM: %s", sim_llm.stats)
    logger.info("Oracle LLM: %s", oracle_llm.stats)

    # 7. Assemble multi-turn SFT data
    sft_path = out_dir / "train.jsonl"
    instances = assemble_sft(out_dir, sft_path)
    logger.info("Final SFT data: %d instances → %s", len(instances), sft_path)

    # 8. Print a sample
    if instances:
        sample = instances[0]
        n_msgs = len(sample["messages"])
        n_user = sum(1 for m in sample["messages"] if m["role"] == "user")
        n_asst = sum(1 for m in sample["messages"] if m["role"] == "assistant")
        logger.info("Sample instance: %d messages (%d user, %d assistant)", n_msgs, n_user, n_asst)
        print("\n" + "=" * 60)
        print("SAMPLE SFT INSTANCE (first conversation):")
        print("=" * 60)
        for m in sample["messages"][:8]:  # Show first 8 messages
            role = m["role"].upper()
            content = m["content"][:200] + ("..." if len(m["content"]) > 200 else "")
            print(f"\n[{role}]\n{content}")
        if n_msgs > 8:
            print(f"\n... ({n_msgs - 8} more messages)")
        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate sample multi-turn SFT data")
    parser.add_argument("--num-scenarios", type=int, default=30,
                        help="Number of persona-grounded scenarios to generate")
    parser.add_argument("--max-turns", type=int, default=12, help="Max conversation turns")
    parser.add_argument("--min-turns", type=int, default=3, help="Min conversation turns")
    parser.add_argument("--persona-id", type=str, default=None,
                        help="Specific persona ID (random if omitted)")
    parser.add_argument("--ablation", type=str, default="full", help="Ablation preset name")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    asyncio.run(generate_sample(
        num_scenarios=args.num_scenarios,
        max_turns=args.max_turns,
        min_turns=args.min_turns,
        persona_id=args.persona_id,
        ablation=args.ablation,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    ))


if __name__ == "__main__":
    main()
