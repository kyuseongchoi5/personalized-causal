"""
Layer 5 & 6: Model Querying + Metrics

Handles API calls to LLMs, answer parsing, and metric computation.
"""

import json
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict
from src.prompts import ExperimentPrompt, PersonaCondition
from src.question_generator import CausalQuestion, QuestionLevel, QuestionCategory


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    model: str
    scenario_name: str
    question_id: str
    condition: str
    question_text: str
    question_level: str
    question_category: str
    gt_answer: str          # "Yes" or "No"
    model_answer: str       # "Yes", "No", or "PARSE_ERROR"
    correct: bool
    raw_response: str
    repetition: int = 0


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_yes_no(response: str) -> str:
    """Extract Yes/No from model response. Returns PARSE_ERROR if ambiguous."""
    text = response.strip().lower()
    # Check for exact match first
    if text in ("yes", "yes."):
        return "Yes"
    if text in ("no", "no."):
        return "No"
    # Check first word
    first_word = text.split()[0] if text.split() else ""
    if first_word.rstrip(".,!:") == "yes":
        return "Yes"
    if first_word.rstrip(".,!:") == "no":
        return "No"
    # Check last word (model may explain then answer)
    words = text.split()
    if words:
        last_word = words[-1].rstrip(".,!:")
        if last_word == "yes":
            return "Yes"
        if last_word == "no":
            return "No"
    # Check for "the answer is yes/no" pattern
    import re
    answer_pattern = re.search(r'\b(answer|conclusion)\b.*?\b(yes|no)\b', text)
    if answer_pattern:
        return "Yes" if answer_pattern.group(2) == "yes" else "No"
    # Check if yes/no appears anywhere (last resort)
    has_yes = "yes" in text
    has_no = "no" in text
    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"
    return "PARSE_ERROR"


# ---------------------------------------------------------------------------
# Model clients
# ---------------------------------------------------------------------------

async def query_openai(
    prompt: ExperimentPrompt,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    # Newer models (gpt-5.5+) require different params
    extra = {}
    if model.startswith("gpt-5"):
        extra["max_completion_tokens"] = 20
        extra["reasoning_effort"] = "none"
    else:
        extra["max_tokens"] = 100
        extra["temperature"] = temperature
    response = await client.chat.completions.create(
        model=model,
        **extra,
        messages=[
            {"role": "system", "content": prompt.system_message},
            {"role": "user", "content": prompt.user_message},
        ],
    )
    return response.choices[0].message.content.strip()


async def query_anthropic(
    prompt: ExperimentPrompt,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.0,
) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=100,
        temperature=temperature,
        system=prompt.system_message,
        messages=[
            {"role": "user", "content": prompt.user_message},
        ],
    )
    return response.content[0].text.strip()


MODEL_CLIENTS = {
    # OpenAI models
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4.1": ("openai", "gpt-4.1"),
    "gpt-4.1-mini": ("openai", "gpt-4.1-mini"),
    "gpt-4.1-nano": ("openai", "gpt-4.1-nano"),
    "gpt-5.5": ("openai", "gpt-5.5"),
    # Anthropic models
    "claude-opus": ("anthropic", "claude-opus-4-20250514"),
    "claude-sonnet": ("anthropic", "claude-sonnet-4-20250514"),
    "claude-haiku": ("anthropic", "claude-haiku-4-5-20251001"),
}


async def query_model(
    prompt: ExperimentPrompt,
    model_name: str,
    temperature: float = 0.0,
) -> str:
    """Dispatch to the right API client based on model name."""
    if model_name not in MODEL_CLIENTS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CLIENTS.keys())}")

    provider, model_id = MODEL_CLIENTS[model_name]
    if provider == "openai":
        return await query_openai(prompt, model=model_id, temperature=temperature)
    elif provider == "anthropic":
        return await query_anthropic(prompt, model=model_id, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_causal_f1(results: list[TrialResult]) -> dict:
    """
    Compute Causal-F1 from a list of trial results.

    Returns precision, recall, F1 overall and broken down by level and category.
    """
    def _f1(results_subset: list[TrialResult]) -> dict:
        if not results_subset:
            return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0, "n": 0}

        # For binary Yes/No, Causal-F1 simplifies.
        # Precision: of answers model said "Yes", how many are truly "Yes"?
        # Recall: of true "Yes" answers, how many did model say "Yes"?
        model_yes = [r for r in results_subset if r.model_answer == "Yes"]
        true_yes = [r for r in results_subset if r.gt_answer == "Yes"]

        tp = sum(1 for r in results_subset if r.model_answer == "Yes" and r.gt_answer == "Yes")
        fp = sum(1 for r in results_subset if r.model_answer == "Yes" and r.gt_answer == "No")
        fn = sum(1 for r in results_subset if r.model_answer == "No" and r.gt_answer == "Yes")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = sum(1 for r in results_subset if r.correct) / len(results_subset)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "n": len(results_subset),
        }

    metrics = {
        "overall": _f1(results),
        "by_level": {},
        "by_category": {},
    }

    # By level
    for level in QuestionLevel:
        subset = [r for r in results if r.question_level == level.value]
        if subset:
            metrics["by_level"][level.value] = _f1(subset)

    # By category
    for cat in QuestionCategory:
        subset = [r for r in results if r.question_category == cat.value]
        if subset:
            metrics["by_category"][cat.value] = _f1(subset)

    return metrics


def compute_drift(
    baseline_results: list[TrialResult],
    condition_results: list[TrialResult],
) -> dict:
    """
    Compute personalization-induced drift between baseline (h0) and another condition.

    Drift = fraction of questions where the answer changed.
    Also broken down by level and category.
    """
    # Match results by question_id
    baseline_map = {r.question_id: r for r in baseline_results}
    condition_map = {r.question_id: r for r in condition_results}

    common_ids = set(baseline_map.keys()) & set(condition_map.keys())

    if not common_ids:
        return {"drift_rate": 0, "n": 0, "flips": []}

    flips = []
    for qid in sorted(common_ids):
        b = baseline_map[qid]
        c = condition_map[qid]
        if b.model_answer != c.model_answer:
            flips.append({
                "question_id": qid,
                "question_text": b.question_text,
                "level": b.question_level,
                "category": b.question_category,
                "gt_answer": b.gt_answer,
                "baseline_answer": b.model_answer,
                "condition_answer": c.model_answer,
            })

    drift_rate = len(flips) / len(common_ids)

    # Drift by level
    drift_by_level = {}
    for level in QuestionLevel:
        level_ids = [qid for qid in common_ids if baseline_map[qid].question_level == level.value]
        if level_ids:
            level_flips = sum(
                1 for qid in level_ids
                if baseline_map[qid].model_answer != condition_map[qid].model_answer
            )
            drift_by_level[level.value] = round(level_flips / len(level_ids), 4)

    # Drift by category
    drift_by_category = {}
    for cat in QuestionCategory:
        cat_ids = [qid for qid in common_ids if baseline_map[qid].question_category == cat.value]
        if cat_ids:
            cat_flips = sum(
                1 for qid in cat_ids
                if baseline_map[qid].model_answer != condition_map[qid].model_answer
            )
            drift_by_category[cat.value] = round(cat_flips / len(cat_ids), 4)

    return {
        "drift_rate": round(drift_rate, 4),
        "n": len(common_ids),
        "n_flips": len(flips),
        "drift_by_level": drift_by_level,
        "drift_by_category": drift_by_category,
        "flips": flips,
    }
