"""
Visualization script for causal world-model drift experiments.
Academic paper style: despined axes, no gridlines, serif fonts, minimal aesthetic.
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
# Constants
# ---------------------------------------------------------------------------

MODEL_ORDER = ["gpt-4o-mini", "gpt-5.5", "claude-haiku", "claude-sonnet", "claude-opus"]
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o\nmini",
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
CONDITION_COLORS = {
    "h0_baseline": "#6baed6",
    "h1_false_belief": "#fdae6b",
    "h2_sycophantic": "#fc9272",
    "h3_neutral_persona": "#74c476",
}

MODEL_COLORS = {
    "gpt-4o-mini": "#1f77b4",
    "gpt-5.5": "#ff7f0e",
    "claude-haiku": "#2ca02c",
    "claude-sonnet": "#d62728",
    "claude-opus": "#9467bd",
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
    "intervention": "Intervention",
    "intervention_hold": "Interv. (hold)",
    "d_separation": "d-Sep.",
    "confounder": "Confounder",
    "mediation": "Mediation",
}


def setup_style():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.labelweight": "bold",
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "figure.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.5,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_root: str) -> pd.DataFrame:
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

def compute_drift_df(df, baseline="h0_baseline", target="h2_sycophantic"):
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

    def classify(row):
        if not row["flipped"]:
            return "no_flip"
        if row["baseline_correct"] and not row["target_correct"]:
            return "correct_to_wrong"
        if not row["baseline_correct"] and row["target_correct"]:
            return "wrong_to_correct"
        return "wrong_to_wrong"

    merged["flip_direction"] = merged.apply(classify, axis=1)
    return merged


# ---------------------------------------------------------------------------
# Fig 1: Accuracy (3-panel: overall, surface, structural)
# ---------------------------------------------------------------------------

def plot_accuracy(df, output_path):
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n = len(models)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.0))

    for ax, (title, filt) in zip(axes, [
        ("(a) Overall", None),
        ("(b) Surface (Type 1)", "surface"),
        ("(c) Structural (Type 2)", "structural"),
    ]):
        sub = df if filt is None else df[df["question_level"] == filt]
        acc = sub.groupby(["model", "condition"])["correct"].mean().reset_index()

        w = 0.16
        x = np.arange(n)
        for i, cond in enumerate(CONDITION_ORDER):
            vals = [
                acc.loc[(acc["model"] == m) & (acc["condition"] == cond), "correct"].values
                for m in models
            ]
            vals = [v[0] if len(v) > 0 else 0 for v in vals]
            ax.bar(x + (i - 1.5) * w, vals, w,
                   color=CONDITION_COLORS[cond], label=CONDITION_LABELS[cond],
                   edgecolor=CONDITION_COLORS[cond], linewidth=0)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=5.5)
        ax.set_ylim(0.38, 1.04)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(title, fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=6.5,
              frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Fig 2: Drift heatmap
# ---------------------------------------------------------------------------

def plot_drift_heatmap(df, output_path):
    drift = compute_drift_df(df)
    pivot = drift.groupby(["model", "scenario_name"])["flipped"].mean().reset_index()

    models = [m for m in MODEL_ORDER if m in pivot["model"].unique()]
    scenarios = sorted(pivot["scenario_name"].unique())

    matrix = pd.DataFrame(0.0, index=models, columns=scenarios)
    for _, row in pivot.iterrows():
        if row["model"] in models:
            matrix.loc[row["model"], row["scenario_name"]] = row["flipped"]

    matrix.index = [MODEL_LABELS_INLINE[m] for m in matrix.index]
    matrix.columns = [SCENARIO_LABELS.get(s, s) for s in matrix.columns]

    fig, ax = plt.subplots(figsize=(6, 2.2))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="YlOrRd",
        vmin=0, vmax=0.4, linewidths=0.6, linecolor="white",
        ax=ax, annot_kws={"fontsize": 6},
        cbar_kws={"shrink": 0.7, "label": "Drift Rate",
                  "aspect": 15},
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40, labelsize=6.5)
    ax.tick_params(axis="y", labelsize=7)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Fig 3 + 5 combined: Surface vs Structural + Flip direction (2-panel)
# ---------------------------------------------------------------------------

def plot_drift_analysis(df, output_path):
    drift = compute_drift_df(df)
    models = [m for m in MODEL_ORDER if m in drift["model"].unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.2))

    # --- Panel (a): Surface vs Structural ---
    x = np.arange(len(models))
    w = 0.3
    for i, (level, label, color) in enumerate([
        ("surface", "Type 1 (Surface)", "#6baed6"),
        ("structural", "Type 2 (Structural)", "#fc9272"),
    ]):
        rates = []
        for m in models:
            s = drift[(drift["model"] == m) & (drift["question_level"] == level)]
            rates.append(s["flipped"].mean() if len(s) > 0 else 0)
        bars = ax1.bar(x + (i - 0.5) * w, rates, w, color=color,
                       label=label, edgecolor=color, linewidth=0)
        for bar, v in zip(bars, rates):
            if v > 0.003:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=5.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=5.5)
    ax1.set_ylabel("Drift Rate")
    ax1.set_ylim(0, max(0.22, ax1.get_ylim()[1] * 1.2))
    ax1.legend(fontsize=6, loc="upper right")
    ax1.set_title("(a) Drift by Question Type", fontsize=8)

    # --- Panel (b): Flip direction ---
    flipped = drift[drift["flipped"]]
    flip_models = [m for m in models if m in flipped["model"].unique()]
    x2 = np.arange(len(flip_models))

    bottoms = np.zeros(len(flip_models))
    for direction, label, color in [
        ("correct_to_wrong", "Correct → Wrong", "#fc9272"),
        ("wrong_to_correct", "Wrong → Correct", "#74c476"),
    ]:
        counts = np.array([
            len(flipped[(flipped["model"] == m) & (flipped["flip_direction"] == direction)])
            for m in flip_models
        ], dtype=float)
        ax2.bar(x2, counts, 0.5, bottom=bottoms, color=color,
                label=label, edgecolor=color, linewidth=0)
        for j, (c, b) in enumerate(zip(counts, bottoms)):
            if c > 0:
                ax2.text(x2[j], b + c / 2, str(int(c)), ha="center", va="center",
                         fontsize=5.5, color="white", fontweight="bold")
        bottoms += counts

    ax2.set_xticks(x2)
    ax2.set_xticklabels([MODEL_LABELS[m] for m in flip_models], fontsize=5.5)
    ax2.set_ylabel("Flipped Answers")
    ax2.legend(fontsize=6, loc="upper right")
    ax2.set_title("(b) Flip Direction", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Fig 4: Category breakdown heatmap
# ---------------------------------------------------------------------------

def plot_category_heatmap(df, output_path):
    drift = compute_drift_df(df)
    models = [m for m in MODEL_ORDER if m in drift["model"].unique()]
    cats = [c for c in CATEGORY_ORDER if c in drift["question_category"].unique()]

    matrix = pd.DataFrame(0.0, index=models, columns=cats)
    for m in models:
        for c in cats:
            s = drift[(drift["model"] == m) & (drift["question_category"] == c)]
            matrix.loc[m, c] = s["flipped"].mean() if len(s) > 0 else 0

    matrix.index = [MODEL_LABELS_INLINE[m] for m in matrix.index]
    matrix.columns = [CATEGORY_LABELS.get(c, c) for c in matrix.columns]

    fig, ax = plt.subplots(figsize=(5, 2.2))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="YlOrRd",
        vmin=0, vmax=0.3, linewidths=0.6, linecolor="white",
        ax=ax, annot_kws={"fontsize": 6},
        cbar_kws={"shrink": 0.7, "label": "Drift Rate", "aspect": 15},
    )
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=6.5)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Fig 5: Condition profiles (small multiples)
# ---------------------------------------------------------------------------

def plot_condition_profiles(df, output_path):
    scenarios = sorted(df["scenario_name"].unique())
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    markers = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "v"]))

    n_cols = 4
    n_rows = (len(scenarios) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 2.2 * n_rows), sharey=True)
    axes_flat = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes_flat[idx]
        for m in models:
            accs = []
            for cond in CONDITION_ORDER:
                s = df[(df["model"] == m) & (df["scenario_name"] == scenario) & (df["condition"] == cond)]
                accs.append(s["correct"].mean() if len(s) > 0 else np.nan)
            ax.plot(range(4), accs, marker=markers[m], markersize=3,
                    color=MODEL_COLORS[m], linewidth=1.0,
                    label=MODEL_LABELS_INLINE[m])

        ax.set_xticks(range(4))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER],
                           rotation=40, ha="right", fontsize=5)
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontsize=7, fontweight="bold")
        ax.set_ylim(0.35, 1.05)
        ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=0.4)
        if idx % n_cols == 0:
            ax.set_ylabel("Accuracy", fontsize=7)

    for idx in range(len(scenarios), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    h, l = axes_flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(models),
              fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
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
    plot_accuracy(df, f"{args.output_dir}/fig1_accuracy.pdf")
    plot_drift_heatmap(df, f"{args.output_dir}/fig2_drift_heatmap.pdf")
    plot_drift_analysis(df, f"{args.output_dir}/fig3_drift_analysis.pdf")
    plot_category_heatmap(df, f"{args.output_dir}/fig4_category_heatmap.pdf")
    plot_condition_profiles(df, f"{args.output_dir}/fig5_condition_profiles.pdf")
    print(f"\nDone. Figures in {args.output_dir}/")


if __name__ == "__main__":
    main()
