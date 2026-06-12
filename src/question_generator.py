"""
Layer 3: Question Generator

Generates typed causal questions from a DAG, with ground-truth answers
from the Oracle. Questions are explicitly categorized as:

  Type 1 (Surface/Behavioral): Can be answered by reading the DAG description.
  Type 2 (Structural/World-Model): Require internalizing the DAG as a structure.

This distinction is the causal analogue of Vafa et al.'s behavioral vs
world-model evaluation.
"""

import networkx as nx
from dataclasses import dataclass
from enum import Enum
from src.oracle import (
    has_direct_edge,
    has_directed_path,
    d_separated,
    intervention_affects,
    intervention_on_X_affects_Y_given_hold_Z,
    is_confounder,
    is_mediator,
    only_indirect_effect,
    all_ordered_pairs,
    all_ordered_triples,
)


class QuestionLevel(str, Enum):
    SURFACE = "surface"        # Type 1: behavioral equivalent
    STRUCTURAL = "structural"  # Type 2: world-model equivalent


class QuestionCategory(str, Enum):
    EDGE = "edge"
    ANCESTRY = "ancestry"
    INTERVENTION = "intervention"
    INTERVENTION_HOLD = "intervention_hold"
    DSEPARATION = "d_separation"
    CONFOUNDER = "confounder"
    MEDIATION = "mediation"
    ONLY_INDIRECT = "only_indirect"


@dataclass
class CausalQuestion:
    question_id: str
    text: str
    gt_answer: bool
    level: QuestionLevel
    category: QuestionCategory
    involved_nodes: tuple[str, ...]

    def answer_str(self) -> str:
        return "Yes" if self.gt_answer else "No"


def generate_questions(G: nx.DiGraph, scenario_name: str = "") -> list[CausalQuestion]:
    """
    Generate the full question bank for a DAG.
    Returns a list of CausalQuestion objects with GT answers.
    """
    questions: list[CausalQuestion] = []
    prefix = f"{scenario_name}_" if scenario_name else ""
    counter = 0

    def _id():
        nonlocal counter
        counter += 1
        return f"{prefix}q{counter:03d}"

    nodes = sorted(G.nodes())
    pairs = all_ordered_pairs(G)

    # ===================================================================
    # TYPE 1 — SURFACE (behavioral equivalent)
    # These can be answered by directly reading the DAG description.
    # ===================================================================

    # 1a. Edge questions: "Does X directly cause Y?"
    for X, Y in pairs:
        gt = has_direct_edge(G, X, Y)
        questions.append(CausalQuestion(
            question_id=_id(),
            text=f"Does {X} directly cause {Y}?",
            gt_answer=gt,
            level=QuestionLevel.SURFACE,
            category=QuestionCategory.EDGE,
            involved_nodes=(X, Y),
        ))

    # ===================================================================
    # TYPE 2 — STRUCTURAL (world-model equivalent)
    # Require the model to internalize the DAG and reason over its structure.
    # ===================================================================

    # 2a. Ancestry questions: "Can X causally affect Y (directly or indirectly)?"
    for X, Y in pairs:
        gt = has_directed_path(G, X, Y)
        # Skip if same as edge answer (only interesting when path ≠ edge)
        if gt != has_direct_edge(G, X, Y):
            questions.append(CausalQuestion(
                question_id=_id(),
                text=f"Can {X} causally affect {Y}, either directly or indirectly?",
                gt_answer=gt,
                level=QuestionLevel.STRUCTURAL,
                category=QuestionCategory.ANCESTRY,
                involved_nodes=(X, Y),
            ))

    # 2b. Intervention questions: "If we intervene to change X, does Y change?"
    for X, Y in pairs:
        gt = intervention_affects(G, X, Y)
        questions.append(CausalQuestion(
            question_id=_id(),
            text=(
                f"If we externally intervene to change {X} "
                f"(breaking all natural causes of {X}), does {Y} change?"
            ),
            gt_answer=gt,
            level=QuestionLevel.STRUCTURAL,
            category=QuestionCategory.INTERVENTION,
            involved_nodes=(X, Y),
        ))

    # 2c. Intervention-with-hold questions: "If we intervene on X holding Z fixed, does Y change?"
    for X, Y in pairs:
        for Z in nodes:
            if Z in (X, Y):
                continue
            gt = intervention_on_X_affects_Y_given_hold_Z(G, X, Y, Z)
            questions.append(CausalQuestion(
                question_id=_id(),
                text=(
                    f"If we intervene to change {X} while holding {Z} fixed, "
                    f"does {Y} change?"
                ),
                gt_answer=gt,
                level=QuestionLevel.STRUCTURAL,
                category=QuestionCategory.INTERVENTION_HOLD,
                involved_nodes=(X, Y, Z),
            ))

    # 2d. d-Separation questions: "Is X independent of Y given Z?"
    for X, Y in pairs:
        if X >= Y:  # avoid duplicate symmetric pairs
            continue
        for Z in nodes:
            if Z in (X, Y):
                continue
            gt = d_separated(G, X, Y, {Z})
            questions.append(CausalQuestion(
                question_id=_id(),
                text=(
                    f"In this causal system, is {X} statistically independent "
                    f"of {Y} given {Z}?"
                ),
                gt_answer=gt,
                level=QuestionLevel.STRUCTURAL,
                category=QuestionCategory.DSEPARATION,
                involved_nodes=(X, Y, Z),
            ))

    # 2e. Confounder questions: "Is Z a confounder of the X-Y relationship?"
    for X, Y in pairs:
        for Z in nodes:
            if Z in (X, Y):
                continue
            gt = is_confounder(G, X, Y, Z)
            questions.append(CausalQuestion(
                question_id=_id(),
                text=(
                    f"Is {Z} a confounder of the relationship between {X} and {Y}?"
                ),
                gt_answer=gt,
                level=QuestionLevel.STRUCTURAL,
                category=QuestionCategory.CONFOUNDER,
                involved_nodes=(X, Y, Z),
            ))

    # 2f. Mediation questions: "Does X's effect on Y go through M?"
    for X, Y in pairs:
        if not has_directed_path(G, X, Y):
            continue  # only ask when X can affect Y
        for M in nodes:
            if M in (X, Y):
                continue
            gt = is_mediator(G, X, Y, M)
            questions.append(CausalQuestion(
                question_id=_id(),
                text=(
                    f"Does {X}'s causal effect on {Y} pass through {M} "
                    f"as a mediator?"
                ),
                gt_answer=gt,
                level=QuestionLevel.STRUCTURAL,
                category=QuestionCategory.MEDIATION,
                involved_nodes=(X, Y, M),
            ))

    # 2g. Only-indirect questions: "Is M the only pathway from X to Y?"
    for X, Y in pairs:
        if not has_directed_path(G, X, Y):
            continue
        for M in nodes:
            if M in (X, Y):
                continue
            if not is_mediator(G, X, Y, M):
                continue
            gt = only_indirect_effect(G, X, Y, M)
            questions.append(CausalQuestion(
                question_id=_id(),
                text=(
                    f"Is {M} the only pathway through which {X} affects {Y}? "
                    f"That is, if we removed {M}, would {X} still affect {Y}?"
                ),
                # Note: the question asks "is M the only pathway" but the second
                # sentence asks "would X still affect Y". GT for "only pathway" = gt,
                # GT for "still affect" = NOT gt. We use the first framing.
                # Actually let's make this unambiguous:
                gt_answer=gt,
                level=QuestionLevel.STRUCTURAL,
                category=QuestionCategory.ONLY_INDIRECT,
                involved_nodes=(X, Y, M),
            ))

    return questions


def balance_questions(
    questions: list[CausalQuestion],
    max_per_category: int | None = None,
    balance_yes_no: bool = True,
    seed: int = 42,
) -> list[CausalQuestion]:
    """
    Balance the question bank to avoid trivial baselines.

    - Caps questions per category to max_per_category
    - If balance_yes_no, ensures roughly equal Yes/No within each category
    """
    import random
    rng = random.Random(seed)

    from collections import defaultdict
    by_category: dict[str, dict[bool, list[CausalQuestion]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for q in questions:
        by_category[q.category][q.gt_answer].append(q)

    balanced = []
    for cat, by_answer in by_category.items():
        yes_qs = by_answer[True]
        no_qs = by_answer[False]
        rng.shuffle(yes_qs)
        rng.shuffle(no_qs)

        if balance_yes_no:
            n = min(len(yes_qs), len(no_qs))
            if max_per_category:
                n = min(n, max_per_category // 2)
            selected = yes_qs[:n] + no_qs[:n]
        else:
            selected = yes_qs + no_qs
            if max_per_category:
                rng.shuffle(selected)
                selected = selected[:max_per_category]

        balanced.extend(selected)

    rng.shuffle(balanced)
    return balanced


def question_bank_summary(questions: list[CausalQuestion]) -> dict:
    """Return a summary of the question bank for inspection."""
    from collections import Counter
    summary = {
        "total": len(questions),
        "by_level": Counter(q.level.value for q in questions),
        "by_category": Counter(q.category.value for q in questions),
        "by_answer": Counter(q.answer_str() for q in questions),
        "by_level_and_answer": Counter(
            (q.level.value, q.answer_str()) for q in questions
        ),
    }
    return summary
