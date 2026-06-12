"""
Layer 1: DAG Library

Each DAG is defined as a dictionary with:
- name: identifier
- domain: subject area
- edges: list of (parent, child) tuples
- description: natural language description of the domain (for Design B, future use)
- false_beliefs: common misconceptions that can seed persona conditions h1
"""

import networkx as nx
from dataclasses import dataclass, field


@dataclass
class CausalScenario:
    name: str
    domain: str
    edges: list[tuple[str, str]]
    description: str
    false_beliefs: list[str] = field(default_factory=list)

    def to_dag(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for parent, child in self.edges:
            G.add_edge(parent, child)
        return G

    def dag_to_text(self) -> str:
        """Convert DAG to explicit natural-language description for Design A prompts."""
        lines = []
        G = self.to_dag()
        nodes = sorted(G.nodes())

        # State direct edges
        for parent, child in sorted(self.edges):
            lines.append(f"- {parent} directly causes {child}")

        # State non-edges (important: explicitly say what is NOT causal)
        for a in nodes:
            for b in nodes:
                if a != b and not G.has_edge(a, b):
                    lines.append(f"- {a} does NOT directly cause {b}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classic textbook scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[CausalScenario] = [
    CausalScenario(
        name="ice_cream_drowning",
        domain="epidemiology",
        edges=[
            ("Temperature", "IceCreamSales"),
            ("Temperature", "Drowning"),
        ],
        description=(
            "Researchers study the relationship between summer temperature, "
            "ice cream sales, and drowning incidents in a coastal city."
        ),
        false_beliefs=[
            "Ice cream sales cause drowning",
            "Drowning causes reduced ice cream sales",
        ],
    ),
    CausalScenario(
        name="smoking_cancer",
        domain="epidemiology",
        edges=[
            ("Smoking", "TarDeposit"),
            ("TarDeposit", "LungCancer"),
            ("Smoking", "LungCancer"),
        ],
        description=(
            "Medical researchers study the relationship between cigarette smoking, "
            "tar deposits in the lungs, and lung cancer incidence."
        ),
        false_beliefs=[
            "Lung cancer causes tar deposits",
            "Tar deposit is unrelated to lung cancer",
        ],
    ),
    CausalScenario(
        name="education_income",
        domain="social_science",
        edges=[
            ("ParentalSES", "Education"),
            ("ParentalSES", "Income"),
            ("Education", "Income"),
        ],
        description=(
            "Economists study the relationship between parental socioeconomic status, "
            "years of education, and adult income."
        ),
        false_beliefs=[
            "Education has no causal effect on income; it's all parental SES",
            "Income determines education level",
        ],
    ),
    CausalScenario(
        name="altitude_health",
        domain="environmental_science",
        edges=[
            ("Altitude", "Temperature"),
            ("Altitude", "OxygenLevel"),
            ("OxygenLevel", "BreathingDifficulty"),
        ],
        description=(
            "Physiologists study the effects of altitude on temperature, "
            "oxygen levels, and breathing difficulty in mountain climbers."
        ),
        false_beliefs=[
            "Low temperature causes breathing difficulty",
            "Breathing difficulty reduces oxygen levels",
        ],
    ),
    CausalScenario(
        name="drug_recovery",
        domain="medicine",
        edges=[
            ("DiseaseSeverity", "DrugAssignment"),
            ("DiseaseSeverity", "Recovery"),
            ("DrugAssignment", "Recovery"),
        ],
        description=(
            "Clinical researchers study how disease severity affects both the "
            "likelihood of receiving a drug and patient recovery outcomes."
        ),
        false_beliefs=[
            "Drug assignment is independent of disease severity",
            "Recovery determines whether a drug is assigned",
        ],
    ),
    CausalScenario(
        name="rain_sprinkler",
        domain="classic_BN",
        edges=[
            ("Rain", "WetGrass"),
            ("Sprinkler", "WetGrass"),
            ("Rain", "Sprinkler"),  # rain discourages sprinkler use
        ],
        description=(
            "A homeowner's lawn can get wet from rain or a sprinkler. "
            "Rain also affects whether the sprinkler is turned on."
        ),
        false_beliefs=[
            "Wet grass causes rain",
            "The sprinkler has no effect on wet grass",
        ],
    ),
    CausalScenario(
        name="exercise_health",
        domain="health",
        edges=[
            ("Exercise", "Fitness"),
            ("Exercise", "HeartHealth"),
            ("Fitness", "HeartHealth"),
            ("Diet", "Fitness"),
            ("Diet", "HeartHealth"),
        ],
        description=(
            "Health researchers study the effects of exercise and diet on "
            "physical fitness and cardiovascular health."
        ),
        false_beliefs=[
            "Heart health causes people to exercise more",
            "Fitness has no effect on heart health",
        ],
    ),
    CausalScenario(
        name="collider_bias",
        domain="statistics",
        edges=[
            ("Talent", "Hollywood"),
            ("Attractiveness", "Hollywood"),
        ],
        description=(
            "Researchers study whether talent and physical attractiveness "
            "independently influence the chance of becoming a Hollywood actor."
        ),
        false_beliefs=[
            "Talent causes attractiveness",
            "Among Hollywood actors, talent and attractiveness are causally related",
        ],
    ),
]


def get_scenario(name: str) -> CausalScenario:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise ValueError(f"Unknown scenario: {name}")


def generate_random_dag(n_nodes: int = 5, edge_prob: float = 0.3, seed: int = 42) -> CausalScenario:
    """Generate a random DAG for synthetic experiments (Design A only)."""
    import random
    rng = random.Random(seed)

    node_names = [f"V{i}" for i in range(n_nodes)]
    edges = []
    # Use topological ordering trick: only allow edges from lower to higher index
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                edges.append((node_names[i], node_names[j]))

    return CausalScenario(
        name=f"synthetic_{seed}",
        domain="synthetic",
        edges=edges,
        description=f"A synthetic causal system with {n_nodes} variables.",
        false_beliefs=[],
    )
