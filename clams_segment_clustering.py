"""
Standalone clustering script for CLAMS Numbers workbook.

Pipeline
--------
1) Load metadata from Metabolic sheet
2) Read mouse sheets 1..80
3) Average all numeric values for each segment within each mouse
4) Build 8-feature vector per mouse
5) Cluster mice into 5 clusters
6) Generate tables + PCA visualizations
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numbers_parser import Document
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# SETTINGS
# ============================================================

N_CLUSTERS = 5
RANDOM_STATE = 42

SEGMENT_ORDER = [
    "ON1", "OFF1",
    "ON2", "OFF2",
    "ON3", "OFF3",
    "ON4", "OFF4"
]

SEGMENT_PATTERN = re.compile(r"^(ON|OFF)\d+$", re.IGNORECASE)

AGE_BINS: List[Tuple[float, float, str]] = [
    (6, 8, "6-8"),
    (12, 14, "12-14"),
    (18, 20, "18-20"),
    (25, 27, "25-27"),
]

AGE_GROUP_ORDER = ["6-8", "12-14", "18-20", "25-27"]


# ============================================================
# AGE GROUP ASSIGNMENT
# ============================================================

def assign_age_group(age_months: float):
    if pd.isna(age_months):
        return np.nan

    for low, high, label in AGE_BINS:
        if low <= float(age_months) <= high:
            return label

    return np.nan


# ============================================================
# NUMBERS SHEET TO DATAFRAME
# ============================================================

def sheet_to_df(sheet):

    if not sheet.tables:
        return pd.DataFrame()

    table = sheet.tables[0]

    rows = []

    for row in table.rows():
        rows.append([getattr(cell, "value", cell) for cell in row])

    if not rows:
        return pd.DataFrame()

    max_len = max(len(r) for r in rows)

    rows = [
        r + [None] * (max_len - len(r))
        for r in rows
    ]

    return pd.DataFrame(rows)


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata(doc):

    meta_sheet = next(
        (s for s in doc.sheets if "metabolic" in s.name.lower()),
        None
    )

    if meta_sheet is None:
        raise ValueError("Metabolic sheet not found.")

    raw = sheet_to_df(meta_sheet)

    meta = raw.iloc[4:, [0, 1, 5]].copy()
    meta.columns = ["mouse", "cohort", "age_months"]

    meta["mouse"] = pd.to_numeric(
        meta["mouse"],
        errors="coerce"
    ).astype("Int64")

    meta = meta.dropna(subset=["mouse"]).copy()

    meta["mouse"] = meta["mouse"].astype(int).astype(str)

    meta["cohort"] = (
        meta["cohort"]
        .astype(str)
        .str.strip()
    )

    meta["age_months"] = pd.to_numeric(
        meta["age_months"],
        errors="coerce"
    )

    meta["age_group"] = meta["age_months"].apply(assign_age_group)

    return meta.reset_index(drop=True)


# ============================================================
# SEGMENT SUMMARY PER MOUSE
# ============================================================

def load_mouse_segment_summary(sheet):

    df = sheet_to_df(sheet)

    if df.empty or df.shape[1] < 3:
        return pd.DataFrame()

    df = df.rename(columns={
        0: "timestamp",
        1: "segment"
    }).copy()

    df["segment"] = (
        df["segment"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[
        df["segment"].str.match(
            SEGMENT_PATTERN,
            na=False
        )
    ].copy()

    feature_cols = [
        c for c in df.columns
        if c not in ("timestamp", "segment")
    ]

    df[feature_cols] = df[feature_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    rows = []

    for seg in SEGMENT_ORDER:

        g = df[df["segment"] == seg]

        values = g[feature_cols].to_numpy(dtype=float).ravel()
        values = values[~np.isnan(values)]

        rows.append({
            "segment": seg,
            "segment_mean": values.mean() if len(values) else np.nan
        })

    return pd.DataFrame(rows)


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

def build_feature_matrix(doc, meta):

    mouse_sheets = [
        s for s in doc.sheets
        if s.name.isdigit() and 1 <= int(s.name) <= 80
    ]

    mouse_sheets = sorted(
        mouse_sheets,
        key=lambda s: int(s.name)
    )

    rows = []

    for sheet in mouse_sheets:

        seg_summary = load_mouse_segment_summary(sheet)

        if seg_summary.empty:
            continue

        row = {"mouse": sheet.name}

        for seg in SEGMENT_ORDER:
            row[seg] = seg_summary.loc[
                seg_summary["segment"] == seg,
                "segment_mean"
            ].iloc[0]

        rows.append(row)

    feat = pd.DataFrame(rows)

    feat = feat.merge(
        meta,
        on="mouse",
        how="left"
    )

    return feat


# ============================================================
# CLUSTERING
# ============================================================

def cluster_mice(feat):

    X = feat[SEGMENT_ORDER].copy()

    X = X.fillna(X.mean())

    X_scaled = StandardScaler().fit_transform(X)

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=50
    )

    feat = feat.copy()

    feat["cluster"] = model.fit_predict(X_scaled)

    return feat


# ============================================================
# TABLES
# ============================================================

def make_tables(clustered):

    valid_age = clustered.dropna(subset=["age_group"]).copy()

    cluster_sizes = (
        clustered.groupby("cluster")
        .size()
        .reset_index(name="n_mice")
        .sort_values("cluster")
    )

    age_dist = pd.crosstab(
        valid_age["cluster"],
        valid_age["age_group"]
    )

    age_dist = age_dist[
        [g for g in AGE_GROUP_ORDER if g in age_dist.columns]
    ]

    cohort_dist = pd.crosstab(
        clustered["cluster"],
        clustered["cohort"]
    ).sort_index(axis=1)

    purity_rows = []

    for cl, g in valid_age.groupby("cluster"):

        counts = g["age_group"].value_counts()

        purity_rows.append({
            "cluster": cl,
            "size": len(g),
            "majority_age_group": counts.idxmax(),
            "purity": counts.max() / counts.sum()
        })

    purity_df = pd.DataFrame(purity_rows)

    overall_purity = sum(
        g["age_group"].value_counts().max()
        for _, g in valid_age.groupby("cluster")
    ) / len(valid_age)

    overall_df = pd.DataFrame([{
        "metric": "overall_age_group_purity",
        "value": overall_purity
    }])

    return {
        "cluster_sizes": cluster_sizes,
        "age_distribution": age_dist,
        "cohort_distribution": cohort_dist,
        "purity_by_cluster": purity_df,
        "overall_purity": overall_df
    }


# ============================================================
# PLOTS
# ============================================================

def plot_cluster_projections(clustered, out_dir):

    X = clustered[SEGMENT_ORDER].copy()
    X = X.fillna(X.mean())

    X_scaled = StandardScaler().fit_transform(X)

    coords = PCA(n_components=2).fit_transform(X_scaled)

    plot_df = clustered.copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]

    plot_specs = [
        (
            "age_group",
            "clusters_colored_by_age_group.png",
            AGE_GROUP_ORDER
        ),
        (
            "cohort",
            "clusters_colored_by_cohort.png",
            sorted(plot_df["cohort"].dropna().unique())
        ),
        (
            "cluster",
            "clusters_colored_by_cluster.png",
            sorted(plot_df["cluster"].unique())
        )
    ]

    for color_col, fname, order in plot_specs:

        plt.figure(figsize=(8, 6))

        for val in order:

            sub = plot_df[
                plot_df[color_col] == val
            ]

            if sub.empty:
                continue

            plt.scatter(
                sub["PC1"],
                sub["PC2"],
                s=70,
                alpha=0.85,
                label=str(val)
            )

            for _, row in sub.iterrows():
                plt.text(
                    row["PC1"],
                    row["PC2"],
                    str(row["mouse"]),
                    fontsize=8
                )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(fname.replace(".png", "").replace("_", " "))
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=300)
        plt.close()


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(clustered, tables, out_dir):

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    clustered.to_csv(
        out_dir / "cluster_assignments.csv",
        index=False
    )

    tables["cluster_sizes"].to_csv(
        out_dir / "cluster_sizes.csv",
        index=False
    )

    tables["age_distribution"].to_csv(
        out_dir / "age_distribution_by_cluster.csv"
    )

    tables["cohort_distribution"].to_csv(
        out_dir / "cohort_distribution_by_cluster.csv"
    )

    tables["purity_by_cluster"].to_csv(
        out_dir / "cluster_purity_by_age_group.csv",
        index=False
    )

    tables["overall_purity"].to_csv(
        out_dir / "overall_purity.csv",
        index=False
    )

    plot_cluster_projections(clustered, out_dir)


# ============================================================
# MAIN
# ============================================================

def main(numbers_file):

    numbers_path = Path(numbers_file)

    doc = Document(str(numbers_path))

    meta = load_metadata(doc)

    feat = build_feature_matrix(doc, meta)

    clustered = cluster_mice(feat)

    tables = make_tables(clustered)

    out_dir = numbers_path.parent / "cluster_outputs"

    save_outputs(clustered, tables, out_dir)

    print("Saved outputs to:", out_dir)


if __name__ == "__main__":

    default_file = "SimplifiedTerskikh10_AgeRoll_CLAMS_081123.numbers"

    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else default_file
    )