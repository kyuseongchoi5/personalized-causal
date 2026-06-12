"""
Layer 2: Causal Oracle

Pure graph-theoretic functions that derive ground-truth answers from a DAG.
No LLM involved. This is the single source of truth.

All functions take a networkx DiGraph and return deterministic boolean answers.
"""

import networkx as nx
from itertools import combinations


# ---------------------------------------------------------------------------
# Primitive graph queries
# ---------------------------------------------------------------------------

def has_direct_edge(G: nx.DiGraph, X: str, Y: str) -> bool:
    """Is there a direct edge X -> Y?"""
    return G.has_edge(X, Y)


def has_directed_path(G: nx.DiGraph, X: str, Y: str) -> bool:
    """Is there any directed path from X to Y (possibly through intermediaries)?"""
    return nx.has_path(G, X, Y)


def ancestors(G: nx.DiGraph, node: str) -> set[str]:
    return nx.ancestors(G, node)


def descendants(G: nx.DiGraph, node: str) -> set[str]:
    return nx.descendants(G, node)


# ---------------------------------------------------------------------------
# d-Separation (the core world-model test)
# ---------------------------------------------------------------------------

def d_separated(G: nx.DiGraph, X: str, Y: str, Z: set[str]) -> bool:
    """
    Test if X and Y are d-separated given Z in DAG G.

    Uses the Bayes-Ball algorithm via networkx.
    X _||_ Y | Z  iff  d_separated returns True.
    """
    return nx.is_d_separator(G, {X}, {Y}, Z)


# ---------------------------------------------------------------------------
# Intervention (do-calculus, mutilated graph)
# ---------------------------------------------------------------------------

def intervention_affects(G: nx.DiGraph, X: str, Y: str) -> bool:
    """
    Does do(X) affect Y?

    Mutilate the graph by removing all incoming edges to X,
    then check if there's a directed path from X to Y.
    """
    G_mut = G.copy()
    incoming = list(G_mut.predecessors(X))
    for parent in incoming:
        G_mut.remove_edge(parent, X)
    return has_directed_path(G_mut, X, Y)


def intervention_on_X_affects_Y_given_hold_Z(
    G: nx.DiGraph, X: str, Y: str, Z: str
) -> bool:
    """
    If we intervene on X while holding Z fixed, does Y change?

    Mutilate: remove incoming edges to X AND to Z (since we fix both),
    then check if X -> Y path exists in the mutilated graph.
    """
    G_mut = G.copy()
    for node in [X, Z]:
        for parent in list(G_mut.predecessors(node)):
            G_mut.remove_edge(parent, node)
    return has_directed_path(G_mut, X, Y)


# ---------------------------------------------------------------------------
# Confounder identification
# ---------------------------------------------------------------------------

def is_confounder(G: nx.DiGraph, X: str, Y: str, Z: str) -> bool:
    """
    Is Z a confounder of the X-Y relationship?

    Z is a confounder if:
    1. Z is an ancestor of X (or Z == X's parent on a backdoor path)
    2. Z is an ancestor of Y
    3. Z is NOT a descendant of X
    4. There is a backdoor path from X to Y through Z

    Simplified: Z has directed paths to both X and Y, and is not a descendant of X.
    """
    if Z == X or Z == Y:
        return False
    return (
        has_directed_path(G, Z, X)
        and has_directed_path(G, Z, Y)
        and not has_directed_path(G, X, Z)
    )


# ---------------------------------------------------------------------------
# Mediation
# ---------------------------------------------------------------------------

def is_mediator(G: nx.DiGraph, X: str, Y: str, M: str) -> bool:
    """
    Does M mediate the effect of X on Y?

    True iff there is a directed path X -> ... -> M -> ... -> Y.
    """
    if M == X or M == Y:
        return False
    return has_directed_path(G, X, M) and has_directed_path(G, M, Y)


def has_direct_effect(G: nx.DiGraph, X: str, Y: str) -> bool:
    """
    Does X have a direct effect on Y (not through any mediator)?
    This is just whether the direct edge X -> Y exists.
    """
    return has_direct_edge(G, X, Y)


def only_indirect_effect(G: nx.DiGraph, X: str, Y: str, M: str) -> bool:
    """
    Is M the ONLY pathway from X to Y?
    True iff: X affects Y, M mediates X->Y, and removing M eliminates all X->Y paths.
    """
    if not is_mediator(G, X, Y, M):
        return False
    G_removed = G.copy()
    G_removed.remove_node(M)
    return not has_directed_path(G_removed, X, Y)


# ---------------------------------------------------------------------------
# Convenience: get all nodes/pairs/triples
# ---------------------------------------------------------------------------

def all_ordered_pairs(G: nx.DiGraph) -> list[tuple[str, str]]:
    nodes = sorted(G.nodes())
    return [(a, b) for a in nodes for b in nodes if a != b]


def all_ordered_triples(G: nx.DiGraph) -> list[tuple[str, str, str]]:
    nodes = sorted(G.nodes())
    return [
        (a, b, c) for a in nodes for b in nodes for c in nodes
        if len({a, b, c}) == 3
    ]
