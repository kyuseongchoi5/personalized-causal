"""
Visualization script for causal world-model drift experiments.

Produces publication-quality figures following academic paper style
(white bg, serif fonts, minimal gridlines, clean axes).

Usage:
    python visualize_results.py
    python visualize_results.py --results-dir results/ --output-dir figures/
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Style & constants
# ---------------------------------------------------------------------------

MODEL_ORDER = ["gpt-4o-mini", "gpt-5.5", "claude-haiku", "claude-sonnet", "claude-opus"]
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5.5": "GPT-5.5",
    "claude-haiku": "Claude\nHaiku",
    "claude-sonnet": "Claude\nSonnet",
    "claude-opus": "Claude\nOpus",
}
MODEL_LABELS_INLINE = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5.5": "GPT-5.5",
    "claude-haiku": "Claude Haiku",
    "claude-sonnet": "Claude Sonnet",
    "claude-opus": "Claude Opus",
}

CONDITION_ORDER = ["h0_baseline", "h1_false_belief", "h2_sycophantic", "h3_neutral_persona"]
CONDITION_LABELS = {
    "h0_baseline": "Baseline",
    "h1_false_belief": "False Belief",
    "h2_sycophantic": "Sycophantic",
    "h3_neutral_persona": "Neutral",
}

# Muted academic palette
PAL = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "red": "#C44E52",
    "green": "#55A868",
    "purple": "#8172B3",
    "gray": "#999999",
}

CONDITION_COLORS = {
    "h0_baseline": PAL["blue"],
    "h1_false_belief": PAL["orange"],
    "h2_sycophantic": PAL["red"],
    "h3_neutral_persona": PAL["green"],
}

MODEL_COLORS = {
    "gpt-4o-mini": PAL["blue"],
    "gpt-5.5": PAL["orange"],
    "claude-haiku": PAL["green"],
    "claude-sonnet": PAL["red"],
    "claude-opus": PAL["purple"],
}

SCENARIO_LABELS = {
    "ice_cream_drowning": "Ice Cream",
    "smoking_cancer": "Smoking",
    "education_income": "Education",
    "climate_agriculture": "Climate",
    "gene_disease": "Gene",
    "social_media_mental_health": "Social Media",
    "supply_chain": "Supply Chain",
    "stress_performance": "Stress",
}

CATEGORY_ORDER = [
    "edge", "intervention", "intervention_hold",
    "d_separation", "confounder", "mediation",
]
CATEGORY_LABELS = {
    "edge": "Edge",
    "ancestry": "Ancestry",
    "intervention": "Intervention",
    "intervention_hold": "Interv. (hold)",
    "d_separation": "d-Sep.",
    "confounder": "Confounder",
    "mediation": "Mediation",
    "only_indirect": "Only Indirect",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "figure.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.5,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_root: str) -> pd.DataFrame:
    """Load and deduplicate all raw_results.jsonl files."""
    all_rows = []
    for path in sorted(Path(results_root).glob("*/raw_results.jsonl")):
        timestamp = path.parent.name
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                row["source_timestamp"] = timestamp
                all_rows.append(row)

    if not all_rows:
        raise FileNotFoundError(f"No raw_results.jsonl found in {results_root}")

    df = pd.DataFrame(all_rows)
    n_before = len(df)
    df = df[df["model_answer"] != "PARSE_ERROR"]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} PARSE_ERROR rows")

    dedup_cols = ["model", "scenario_name", "question_id", "condition", "repetition"]
    df = df.sort_values("source_timestamp").drop_duplicates(subset=dedup_cols, keep="last")
    df = df[df["model"].isin(MODEL_ORDER)]

    print(f"  Loaded {len(df)} valid trials")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Scenarios: {sorted(df['scenario_name'].unique())}")

    return df


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------

def compute_drift_df(
    df: pd.DataFrame,
    baseline: str = "h0_baseline",
    target: str = "h2_sycophantic",
) -> pd.DataFrame:
    base = df[df["condition"] == baseline].set_index(
        ["model", "scenario_name", "question_id"]
    )[["model_answer", "gt_answer", "question_level", "question_category", "correct"]].rename(
        columns={"model_answer": "baseline_answer", "correct": "baseline_correct"}
    )
    tgt = df[df["condition"] == target].set_index(
        ["model", "scenario_name", "question_id"]
    )[["model_answer", "correct"]].rename(
        columns={"model_answer": "target_answer", "correct": "target_correct"}
    )

    merged = base.join(tgt, how="inner").reset_index()
    merged["flipped"] = merged["baseline_answer"] != merged["target_answer"]

    def classify_flip(row):
        if not row["flipped"]:
            return "no_flip"
        if row["baseline_correct"] and not row["target_correct"]:
            return "correct_to_wrong"
        if not row["baseline_correct"] and row["target_correct"]:
            return "wrong_to_correct"
        return "wrong_to_wrong"

    merged["flip_direction"] = merged.apply(classify_flip, axis=1)
    return merged


# ---------------------------------------------------------------------------
# Plot 1: Accuracy by Model x Condition (3-panel: overall, surface, structural)
# ---------------------------------------------------------------------------

def plot_accuracy_by_model_condition(df: pd.DataFrame, output_path: str):
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n_models = len(models)
    n_conds = len(CONDITION_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    titles = ["(a) Overall Accuracy", "(b) Surface (Type 1)", "(c) Structural (Type 2)"]
    filters = [None, "surface", "structural"]

    for ax, title, level_filter in zip(axes, titles, filters):
        sub = df if level_filter is None else df[df["question_level"] == level_filter]
        acc = sub.groupby(["model", "condition"])["correct"].mean().reset_index()
        acc.columns = ["model", "condition", "accuracy"]

        bar_width = 0.17
        x = np.arange(n_models)

        for i, cond in enumerate(CONDITION_ORDER):
            vals = []
            for m in models:
                v = acc[(acc["model"] == m) & (acc["condition"] == cond)]["accuracy"]
                vals.append(v.values[0] if len(v) > 0 else 0)
            offset = (i - n_conds / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, vals, bar_width,
                label=CONDITION_LABELS[cond],
                color=CONDITION_COLORS[cond],
                edgecolor="white", linewidth=0.3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=7)
        ax.set_ylim(0.35, 1.08)
        ax.axhline(0.5, color=PAL["gray"], linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_title(title, fontsize=9, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_conds,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Drift Heatmap
# ---------------------------------------------------------------------------

def plot_drift_heatmap(df: pd.DataFrame, output_path: str):
    drift = compute_drift_df(df)
    pivot = drift.groupby(["model", "scenario_name"])["flipped"].mean().reset_index()
    pivot.columns = ["model", "scenario", "drift_rate"]

    models = [m for m in MODEL_ORDER if m in pivot["model"].unique()]
    scenarios = sorted(pivot["scenario"].unique())

    matrix = pd.DataFrame(index=models, columns=scenarios, dtype=float).fillna(0)
    for _, row in pivot.iterrows():
        if row["model"] in models:
            matrix.loc[row["model"], row["scenario"]] = row["drift_rate"]

    matrix.index = [MODEL_LABELS_INLINE.get(m, m) for m in matrix.index]
    matrix.columns = [SCENARIO_LABELS.get(s, s) for s in matrix.columns]

    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 0.95), 2.8))
    sns.heatmap(
        matrix.astype(float), annot=True, fmt=".2f",
        cmap="YlOrRd", vmin=0, vmax=0.45,
        linewidths=0.8, linecolor="white", ax=ax,
        annot_kws={"fontsize": 7, "fontweight": "bold"},
        cbar_kws={"label": "Drift Rate", "shrink": 0.8},
    )
    ax.set_title("Personalization-Induced Drift Rate (Sycophantic vs. Baseline)",
                 fontsize=9, fontweight="bold", pad=8)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Surface vs Structural Drift
# ---------------------------------------------------------------------------

def plot_surface_vs_structural(df: pd.DataFrame, output_path: str):
    drift = compute_drift_df(df)
    models = [m for m in MODEL_ORDER if m in drift["model"].unique()]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(models))
    bar_width = 0.28

    for i, (level, label, color) in enumerate([
        ("surface", "Type 1 (Surface)", PAL["blue"]),
        ("structural", "Type 2 (Structural)", PAL["red"]),
    ]):
        rates = []
        for m in models:
            subset = drift[(drift["model"] == m) & (drift["question_level"] == level)]
            rates.append(subset["flipped"].mean() if len(subset) > 0 else 0)
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, rates, bar_width, label=label,
                      color=color, edgecolor="white", linewidth=0.3)
        for bar, v in zip(bars, rates):
            if v > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=7)
    ax.set_ylabel("Drift Rate")
    ax.set_ylim(0, max(0.25, ax.get_ylim()[1] * 1.18))
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("(a) Drift by Question Type", fontsize=9, fontweight="bold")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 4: Category Breakdown
# ---------------------------------------------------------------------------

def plot_category_vulnerability(df: pd.DataFrame, output_path: str):
    drift = compute_drift_df(df)
    models = [m for m in MODEL_ORDER if m in drift["model"].unique()]
    categories = [c for c in CATEGORY_ORDER if c in drift["question_category"].unique()]

    matrix = pd.DataFrame(index=models, columns=categories, dtype=float)
    for m in models:
        for c in categories:
            subset = drift[(drift["model"] == m) & (drift["question_category"] == c)]
            matrix.loc[m, c] = subset["flipped"].mean() if len(subset) > 0 else 0

    matrix.index = [MODEL_LABELS_INLINE.get(m, m) for m in matrix.index]
    matrix.columns = [CATEGORY_LABELS.get(c, c) for c in matrix.columns]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    sns.heatmap(
        matrix.astype(float), annot=True, fmt=".2f",
        cmap="YlOrRd", vmin=0, vmax=0.35,
        linewidths=0.8, linecolor="white", ax=ax,
        annot_kws={"fontsize": 7, "fontweight": "bold"},
        cbar_kws={"label": "Drift Rate", "shrink": 0.8},
    )
    ax.set_title("Drift Rate by Question Category (Sycophantic Condition)",
                 fontsize=9, fontweight="bold", pad=8)
    ax.set_ylabel("")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 5: Flip Direction Analysis
# ---------------------------------------------------------------------------

def plot_flip_direction(df: pd.DataFrame, output_path: str):
    drift = compute_drift_df(df)
    flipped = drift[drift["flipped"]]

    if flipped.empty:
        print("  No flips found -- skipping")
        return

    models = [m for m in MODEL_ORDER if m in flipped["model"].unique()]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(models))

    directions = ["correct_to_wrong", "wrong_to_correct"]
    dir_labels = {"correct_to_wrong": r"Correct $\rightarrow$ Wrong",
                  "wrong_to_correct": r"Wrong $\rightarrow$ Correct"}
    dir_colors = {"correct_to_wrong": PAL["red"], "wrong_to_correct": PAL["green"]}

    bottoms = np.zeros(len(models))
    for direction in directions:
        counts = np.array([
            len(flipped[(flipped["model"] == m) & (flipped["flip_direction"] == direction)])
            for m in models
        ], dtype=float)
        ax.bar(x, counts, 0.5, bottom=bottoms,
               label=dir_labels[direction], color=dir_colors[direction],
               edgecolor="white", linewidth=0.3)
        for j, (c, b) in enumerate(zip(counts, bottoms)):
            if c > 0:
                ax.text(x[j], b + c / 2, str(int(c)), ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=7)
    ax.set_ylabel("Number of Flipped Answers")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("(b) Direction of Answer Flips", fontsize=9, fontweight="bold")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 6: Condition Profiles (small multiples)
# ---------------------------------------------------------------------------

def plot_condition_profiles(df: pd.DataFrame, output_path: str):
    scenarios = sorted(df["scenario_name"].unique())
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    n_cols = 4
    n_rows = (len(scenarios) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.8 * n_rows), sharey=True)
    axes = axes.flatten() if n_rows > 1 else (axes if len(scenarios) > 1 else [axes])

    model_markers = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "v"]))

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        for m in models:
            accs = []
            for cond in CONDITION_ORDER:
                subset = df[(df["model"] == m) & (df["scenario_name"] == scenario) & (df["condition"] == cond)]
                accs.append(subset["correct"].mean() if len(subset) > 0 else np.nan)
            ax.plot(
                range(len(CONDITION_ORDER)), accs,
                marker=model_markers.get(m, "o"), markersize=4,
                label=MODEL_LABELS_INLINE.get(m, m),
                color=MODEL_COLORS.get(m), linewidth=1.2,
            )

        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels(
            [CONDITION_LABELS[c] for c in CONDITION_ORDER],
            rotation=35, ha="right", fontsize=6.5,
        )
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontsize=8, fontweight="bold")
        ax.set_ylim(0.35, 1.05)
        ax.axhline(0.5, color=PAL["gray"], linestyle="--", linewidth=0.5, alpha=0.4)
        if idx % n_cols == 0:
            ax.set_ylabel("Accuracy", fontsize=8)

    # Hide unused subplots
    for idx in range(len(scenarios), len(axes)):
        axes[idx].set_visible(False)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output-dir", default="figures/")
    args = parser.parse_args()

    setup_style()

    print("Loading results...")
    df = load_all_results(args.results_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating figures...")
    plot_accuracy_by_model_condition(df, f"{args.output_dir}/fig1_accuracy_by_model_condition.pdf")
    plot_drift_heatmap(df, f"{args.output_dir}/fig2_drift_heatmap.pdf")
    plot_surface_vs_structural(df, f"{args.output_dir}/fig3_surface_vs_structural.pdf")
    plot_category_vulnerability(df, f"{args.output_dir}/fig4_category_breakdown.pdf")
    plot_flip_direction(df, f"{args.output_dir}/fig5_flip_direction.pdf")
    plot_condition_profiles(df, f"{args.output_dir}/fig6_condition_profiles.pdf")

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
