"""
analyze_mesh_jaccard.py

Compare average Jaccard distance between source articles' MeSH terms and those
of their citing articles vs year-matched random controls, across multiple fields.
"""

import ast

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns

from artificial_intelligence_in_medicine.config import INTERIM_DATA_DIR


def to_set(x):
    if isinstance(x, str):
        try:
            return set(ast.literal_eval(x))
        except Exception:
            return set([t.strip() for t in x.split("|") if t.strip()])
    return set(x) if isinstance(x, (list, set)) else set()


def jaccard_distance(a, b):
    a, b = to_set(a), to_set(b)
    if not a and not b:
        return 1.0
    return 1.0 - len(a & b) / len(a | b)


def compute_citing_means(df, mesh_lookup):
    rows = []
    for _, row in df.iterrows():
        src_id = row["pmid"]
        src_mesh = row["mesh_headings"]
        cited_pmids = row["cited_by"] or []
        cited_pmids = [c for c in cited_pmids if c in mesh_lookup]
        if not cited_pmids:
            continue
        dists = [jaccard_distance(src_mesh, mesh_lookup[c]) for c in cited_pmids]
        rows.append({"source": src_id, "cite_mean": np.mean(dists), "n_citing": len(dists)})
    return pd.DataFrame(rows)


def compute_control_means(df, mesh_lookup, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    all_pmids = list(mesh_lookup.keys())
    rows = []
    for _, row in df.iterrows():
        src_id = row["pmid"]
        src_mesh = row["mesh_headings"]
        n_citing = len(row["cited_by"] or [])
        if n_citing == 0:
            continue
        pool = [p for p in all_pmids if p != src_id]
        if len(pool) < n_citing:
            continue
        sample_pmids = rng.choice(pool, size=n_citing, replace=False)
        dists = [jaccard_distance(src_mesh, mesh_lookup[c]) for c in sample_pmids]
        rows.append({"source": src_id, "ctrl_mean": np.mean(dists)})
    return pd.DataFrame(rows)


def run_analysis(df, label):
    mesh_lookup = dict(zip(df["pmid"], df["mesh_headings"]))
    citing_df = compute_citing_means(df, mesh_lookup)
    control_df = compute_control_means(df, mesh_lookup)

    merged = pd.merge(citing_df, control_df, on="source", how="inner")
    merged["delta"] = merged["cite_mean"] - merged["ctrl_mean"]
    merged["field"] = label

    stat, pval = wilcoxon(merged["delta"], alternative="less")
    median_delta = merged["delta"].median()

    print(f"\n=== Results: {label} ===")
    print(f"Number of sources tested: {len(merged)}")
    print(f"Wilcoxon signed-rank statistic: {stat}")
    print(f"One-sided p-value (citing < control): {pval:.4g}")
    print(f"Median delta: {median_delta:.4f}")

    return merged


def plot_combined(df_all):
    # Prepare the data for plotting
    plot_df = pd.DataFrame(
        {
            "Citing": citing_means,
            "Control": control_means[: len(citing_means)],  # match lengths
        }
    ).melt(var_name="Group", value_name="Mean Jaccard Distance")

    plt.figure(figsize=(8, 6))

    # Violin plot
    sns.violinplot(
        data=plot_df,
        x="Group",
        y="Mean Jaccard Distance",
        inner="box",  # show boxplot inside
        palette={"Citing": "skyblue", "Control": "salmon"},
        cut=0,
    )

    # Add reference line at global median
    global_median = plot_df["Mean Jaccard Distance"].median()
    plt.axhline(global_median, color="gray", linestyle="--", alpha=0.7)

    # Annotate median delta
    median_cite = np.median(citing_means)
    median_ctrl = np.median(control_means[: len(citing_means)])
    delta = median_cite - median_ctrl
    plt.text(
        0.5,
        max(plot_df["Mean Jaccard Distance"]) * 0.95,
        f"Median Δ = {delta:.3f}",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    plt.title("Citing vs. Control: Jaccard Distance of MeSH Terms", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    datasets = {
        "Artificial Intelligence": INTERIM_DATA_DIR
        / "ARTIFICIAL_INTELLIGENCE"
        / "features_with_ror.json",
        "Gene Expression": INTERIM_DATA_DIR / "GENE_EXPRESSION" / "features_with_ror.json",
    }

    results_all = []
    for label, path in datasets.items():
        df = pd.read_json(path)
        print(f"\n--- {label} ---")
        print("Total rows:", len(df))
        print("Unique PMIDs:", df["pmid"].nunique())
        print("Rows with MeSH:", df["mesh_headings"].notna().sum())
        print(
            "Rows with citing articles:",
            df["cited_by"].map(lambda x: len(x) if isinstance(x, list) else 0).gt(0).sum(),
        )
        results_all.append(run_analysis(df, label))

    combined = pd.concat(results_all, ignore_index=True)
    plot_combined(combined)
