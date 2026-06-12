"""
Main experiment runner.

Usage:
    python run_experiment.py --models gpt-4o claude-sonnet --scenarios ice_cream_drowning smoking_cancer
    python run_experiment.py --all  # run all scenarios and models
    python run_experiment.py --dry-run  # print question bank stats without calling APIs
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.dag_library import SCENARIOS, get_scenario, CausalScenario
from src.question_generator import (
    generate_questions,
    balance_questions,
    question_bank_summary,
    CausalQuestion,
)
from src.prompts import build_prompt, PersonaCondition
from src.evaluator import (
    query_model,
    parse_yes_no,
    TrialResult,
    compute_causal_f1,
    compute_drift,
    MODEL_CLIENTS,
)


CONDITIONS = [
    PersonaCondition.BASELINE,
    PersonaCondition.FALSE_BELIEF,
    PersonaCondition.SYCOPHANTIC,
    PersonaCondition.NEUTRAL_PERSONA,
]


async def run_single_trial(
    model_name: str,
    scenario: CausalScenario,
    question: CausalQuestion,
    condition: PersonaCondition,
    temperature: float = 0.0,
    repetition: int = 0,
) -> TrialResult:
    """Run a single (model, scenario, question, condition) trial."""
    prompt = build_prompt(scenario, question, condition)

    try:
        raw = await query_model(prompt, model_name, temperature=temperature)
    except Exception as e:
        raw = f"ERROR: {e}"

    model_answer = parse_yes_no(raw)
    gt = question.answer_str()

    return TrialResult(
        model=model_name,
        scenario_name=scenario.name,
        question_id=question.question_id,
        condition=condition.value,
        question_text=question.text,
        question_level=question.level.value,
        question_category=question.category.value,
        gt_answer=gt,
        model_answer=model_answer,
        correct=(model_answer == gt),
        raw_response=raw,
        repetition=repetition,
    )


async def run_experiment(
    model_names: list[str],
    scenarios: list[CausalScenario],
    conditions: list[PersonaCondition],
    max_questions_per_category: int | None = 10,
    balance: bool = True,
    temperature: float = 0.0,
    repetitions: int = 1,
    concurrency: int = 10,
) -> list[TrialResult]:
    """Run the full experiment across all combinations."""
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_trial(*args, **kwargs):
        async with semaphore:
            return await run_single_trial(*args, **kwargs)

    # Build all tasks
    tasks = []
    for scenario in scenarios:
        G = scenario.to_dag()
        questions = generate_questions(G, scenario.name)

        if balance or max_questions_per_category:
            questions = balance_questions(
                questions,
                max_per_category=max_questions_per_category,
                balance_yes_no=balance,
            )

        print(f"\n[{scenario.name}] Question bank:")
        summary = question_bank_summary(questions)
        print(f"  Total: {summary['total']}")
        print(f"  By level: {dict(summary['by_level'])}")
        print(f"  By category: {dict(summary['by_category'])}")
        print(f"  By answer: {dict(summary['by_answer'])}")

        for model_name in model_names:
            for condition in conditions:
                for question in questions:
                    for rep in range(repetitions):
                        tasks.append(
                            bounded_trial(
                                model_name, scenario, question,
                                condition, temperature, rep,
                            )
                        )

    total = len(tasks)
    print(f"\nTotal API calls: {total}")
    print(f"Models: {model_names}")
    print(f"Conditions: {[c.value for c in conditions]}")
    print(f"Starting experiment...\n")

    results = await asyncio.gather(*tasks)
    return list(results)


def analyze_results(results: list[TrialResult]) -> dict:
    """Compute all metrics from experiment results."""
    analysis = {}

    # Group by (model, scenario)
    groups = {}
    for r in results:
        key = (r.model, r.scenario_name)
        if key not in groups:
            groups[key] = {}
        if r.condition not in groups[key]:
            groups[key][r.condition] = []
        groups[key][r.condition].append(r)

    for (model, scenario), by_condition in groups.items():
        analysis_key = f"{model}__{scenario}"
        analysis[analysis_key] = {"causal_f1": {}, "drift": {}}

        # Causal-F1 per condition
        for cond, cond_results in by_condition.items():
            analysis[analysis_key]["causal_f1"][cond] = compute_causal_f1(cond_results)

        # Drift: h0 vs each other condition
        baseline_key = PersonaCondition.BASELINE.value
        if baseline_key in by_condition:
            for cond, cond_results in by_condition.items():
                if cond == baseline_key:
                    continue
                analysis[analysis_key]["drift"][f"{baseline_key}_vs_{cond}"] = compute_drift(
                    by_condition[baseline_key], cond_results,
                )

    return analysis


def print_summary(analysis: dict):
    """Print a human-readable summary of results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)

    for key, data in analysis.items():
        model, scenario = key.split("__")
        print(f"\n{'─' * 60}")
        print(f"Model: {model}  |  Scenario: {scenario}")
        print(f"{'─' * 60}")

        # Causal-F1
        print("\n  Causal-F1 by condition:")
        for cond, metrics in data["causal_f1"].items():
            overall = metrics["overall"]
            surface = metrics["by_level"].get("surface", {})
            structural = metrics["by_level"].get("structural", {})
            print(f"    {cond}:")
            print(f"      Overall    — Acc: {overall['accuracy']:.3f}  F1: {overall['f1']:.3f}  (n={overall['n']})")
            if surface:
                print(f"      Surface    — Acc: {surface['accuracy']:.3f}  F1: {surface['f1']:.3f}  (n={surface['n']})")
            if structural:
                print(f"      Structural — Acc: {structural['accuracy']:.3f}  F1: {structural['f1']:.3f}  (n={structural['n']})")

        # Drift
        if data["drift"]:
            print("\n  Drift (vs baseline):")
            for comparison, drift in data["drift"].items():
                cond_name = comparison.replace(f"{PersonaCondition.BASELINE.value}_vs_", "")
                print(f"    → {cond_name}: {drift['drift_rate']:.3f} ({drift['n_flips']}/{drift['n']} flipped)")
                if drift.get("drift_by_level"):
                    for level, rate in drift["drift_by_level"].items():
                        print(f"        {level}: {rate:.3f}")


def save_results(results: list[TrialResult], analysis: dict, output_dir: str):
    """Save raw results and analysis to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Raw results as JSONL
    with open(f"{output_dir}/raw_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r.__dict__) + "\n")

    # Analysis as JSON
    with open(f"{output_dir}/analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def dry_run(scenarios: list[CausalScenario], max_per_cat: int | None = 10):
    """Print question bank stats without calling any API."""
    print("DRY RUN — Question bank statistics\n")
    for scenario in scenarios:
        G = scenario.to_dag()
        raw = generate_questions(G, scenario.name)
        balanced = balance_questions(raw, max_per_category=max_per_cat)

        print(f"{'─' * 50}")
        print(f"Scenario: {scenario.name} ({len(G.nodes())} nodes, {len(G.edges())} edges)")
        print(f"  Nodes: {sorted(G.nodes())}")
        print(f"  Edges: {sorted(G.edges())}")
        print(f"\n  Raw questions: {len(raw)}")
        raw_summary = question_bank_summary(raw)
        print(f"    By level: {dict(raw_summary['by_level'])}")
        print(f"    By category: {dict(raw_summary['by_category'])}")
        print(f"    By answer: {dict(raw_summary['by_answer'])}")
        print(f"\n  Balanced questions: {len(balanced)}")
        bal_summary = question_bank_summary(balanced)
        print(f"    By level: {dict(bal_summary['by_level'])}")
        print(f"    By category: {dict(bal_summary['by_category'])}")
        print(f"    By answer: {dict(bal_summary['by_answer'])}")

        # Show a few example questions
        print(f"\n  Sample questions:")
        for q in balanced[:5]:
            print(f"    [{q.level.value:10s}] [{q.category.value:18s}] {q.text}  → GT: {q.answer_str()}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Causal World-Model Drift Experiment")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"],
                        help=f"Models to test. Available: {list(MODEL_CLIENTS.keys())}")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Scenario names. Default: all.")
    parser.add_argument("--max-per-category", type=int, default=10,
                        help="Max questions per category after balancing")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print question stats without API calls")
    parser.add_argument("--all", action="store_true",
                        help="Run all scenarios and default models")

    args = parser.parse_args()

    # Select scenarios
    if args.scenarios:
        scenarios = [get_scenario(s) for s in args.scenarios]
    elif args.all:
        scenarios = SCENARIOS
    else:
        scenarios = SCENARIOS[:3]  # default: first 3

    if args.dry_run:
        dry_run(scenarios, args.max_per_category)
        return

    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/{timestamp}"

    # Run
    results = asyncio.run(run_experiment(
        model_names=args.models,
        scenarios=scenarios,
        conditions=CONDITIONS,
        max_questions_per_category=args.max_per_category,
        temperature=args.temperature,
        repetitions=args.repetitions,
        concurrency=args.concurrency,
    ))

    # Analyze
    analysis = analyze_results(results)
    print_summary(analysis)
    save_results(results, analysis, args.output_dir)


if __name__ == "__main__":
    main()
