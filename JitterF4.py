# ==========================================================
# RER DISTANCE TO MOUSE #1 + UMAP + COLORED AGE REGIONS
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from numbers_parser import Document
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import umap

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
FILE_PATH = "SimplifiedTerskikh10_AgeRoll_CLAMS_081123.numbers"
REF_MOUSE = "1"
TARGET_N_OLD = 20

AGE_GROUPS = {
    "6-8": (6, 8),
    "12-14": (12, 14),
    "17-20": (17, 20),
    "24-28": (24, 28),
}
AGE_ORDER = list(AGE_GROUPS.keys())

COLORS = {
    "6-8": "#1f77b4",
    "12-14": "#2ca02c",
    "17-20": "#ff7f0e",
    "24-28": "#d62728",
}

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
def load_numbers_file(path):
    doc = Document(path)
    sheets = {}

    for sheet in doc.sheets:
        if not sheet.tables:
            continue

        table = sheet.tables[0]
        data = [[getattr(cell, "value", None) for cell in row] for row in table.rows()]

        if data:
            sheets[sheet.name] = pd.DataFrame(data)

    return sheets

def preprocess_metadata(df):
    df = df.iloc[4:].copy()
    df = df[df[0].notna()]

    df = df.rename(columns={0: "mouse_label", 5: "age_months"})
    df["mouse_label"] = pd.to_numeric(df["mouse_label"], errors="coerce").astype("Int64").astype(str)
    df["age_months"] = pd.to_numeric(df["age_months"], errors="coerce")

    return df[["mouse_label", "age_months"]]

def load_numeric(df):
    df = df.copy()
    df["segment"] = df.iloc[:, 1].astype(str).str.strip().str.upper()
    df = df.apply(pd.to_numeric, errors="coerce").ffill().fillna(0)
    return df

# -------------------------------------------------
# DISTANCE FUNCTION
# -------------------------------------------------
def compute_segment_distance(df, df_ref, segments, var_name):
    distances = []

    for seg in segments:
        ts = df[df["segment"] == seg][var_name].dropna().values
        ref_ts = df_ref[df_ref["segment"] == seg][var_name].dropna().values

        if len(ts) == 0 or len(ref_ts) == 0:
            distances.append(1.0)
            continue

        min_len = min(len(ts), len(ref_ts))
        x, y = ref_ts[:min_len], ts[:min_len]

        if np.std(x) == 0 or np.std(y) == 0:
            distances.append(1.0)
            continue

        corr, _ = pearsonr(x, y)
        distances.append(1 - corr if not np.isnan(corr) else 1.0)

    return distances

# -------------------------------------------------
# AGE GROUP
# -------------------------------------------------
def assign_age_group(age):
    if pd.isna(age):
        return "Unknown"
    for g, (low, high) in AGE_GROUPS.items():
        if low <= age <= high:
            return g
    return "Other"

# -------------------------------------------------
# LOAD EVERYTHING
# -------------------------------------------------
sheets = load_numbers_file(FILE_PATH)

meta_sheet = next(s for s in sheets if "metabolic" in s.lower())
mouse_sheets = sorted([s for s in sheets if s.strip().isdigit()])

meta = preprocess_metadata(sheets[meta_sheet])

variable_matrices = {
    m: load_numeric(sheets[m]) for m in mouse_sheets
}

# variable
sample_df = next(iter(variable_matrices.values()))
var_name = [c for c in sample_df.columns if c != "segment"][4]
print("Using variable:", var_name)

# segments
df_ref = variable_matrices[REF_MOUSE]
all_segments = sorted(set(seg for df in variable_matrices.values() for seg in df["segment"]))
selected_segments = all_segments[:8]

# -------------------------------------------------
# DISTANCE TABLE
# -------------------------------------------------
results = []
for mouse, df in variable_matrices.items():
    d = compute_segment_distance(df, df_ref, selected_segments, var_name)
    results.append({"mouse": mouse, "distance": np.mean(d)})

distance_df = pd.DataFrame(results)
distance_df = distance_df.merge(meta, left_on="mouse", right_on="mouse_label", how="left")
distance_df["age_group"] = distance_df["age_months"].apply(assign_age_group)

distance_df["age_group"] = pd.Categorical(distance_df["age_group"], categories=AGE_ORDER, ordered=True)
plot_df = distance_df[distance_df["age_group"].isin(AGE_ORDER)].copy()

# -------------------------------------------------
# BALANCE OLD GROUP
# -------------------------------------------------
young = plot_df[plot_df["age_group"] != "24-28"]
old = plot_df[plot_df["age_group"] == "24-28"]
old = old.sample(n=TARGET_N_OLD, replace=True, random_state=42)

plot_df = pd.concat([young, old])

# -------------------------------------------------
# UMAP FEATURES
# -------------------------------------------------
features = []
for mouse in plot_df["mouse"]:
    df = variable_matrices[mouse]
    d = compute_segment_distance(df, df_ref, selected_segments, var_name)
    features.append(d)

X = StandardScaler().fit_transform(np.nan_to_num(features, nan=1.0))

embedding = umap.UMAP(
    n_neighbors=10,
    min_dist=0.25,
    random_state=42
).fit_transform(X)

umap_df = plot_df.copy()
umap_df["UMAP1"] = embedding[:, 0]
umap_df["UMAP2"] = embedding[:, 1]

# -------------------------------------------------
# FINAL PLOT (COLORED NONLINEAR REGIONS)
# -------------------------------------------------
def plot_umap_colored_regions(umap_df):

    Xp = umap_df[["UMAP1", "UMAP2"]].values
    y_cat = umap_df["age_group"]
    y = y_cat.cat.codes

    clf = SVC(kernel="rbf", C=1)
    clf.fit(Xp, y)

    xx, yy = np.meshgrid(
        np.linspace(Xp[:,0].min()-0.5, Xp[:,0].max()+0.5, 400),
        np.linspace(Xp[:,1].min()-0.5, Xp[:,1].max()+0.5, 400)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(9,6))

    # background regions
    for i, g in enumerate(y_cat.cat.categories):
        mask = (Z == i)
        plt.contourf(xx, yy, mask, levels=[0.5, 1], colors=[COLORS[g]], alpha=0.25)

    # points
    for g in AGE_ORDER:
        sub = umap_df[umap_df["age_group"] == g]
        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            label=g,
            color=COLORS[g],
            edgecolor="black",
            s=50
        )

    plt.legend(title="Age Group")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP with Curved Age Regions")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return clf, Xp, y

clf, Xp, y = plot_umap_colored_regions(umap_df)

# -------------------------------------------------
# VALIDATION
# -------------------------------------------------
score = cross_val_score(clf, Xp, y, cv=5).mean()
print("Cross-val accuracy:", score)