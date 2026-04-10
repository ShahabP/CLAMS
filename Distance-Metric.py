
# -*- coding: utf-8 -*-
"""
ROBUST CLAMS CLUSTERING PIPELINE

Features
--------
• Supports .xls .xlsx .numbers
• Automatic optimal cluster selection
• Multiview clustering
• Wavelet pattern recognition
• PCA visualization
• Time-series clustering
• Segment-aware plots
• Robust variable-length handling
"""

# -------------------------------------------------
# Imports
# -------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

from mvlearn.cluster import MultiviewKMeans

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import pywt

os.environ["OMP_NUM_THREADS"] = "1"


# -------------------------------------------------
# File path
# -------------------------------------------------
file_path = os.path.join(
    os.path.dirname(__file__),
    "SimplifiedTerskikh10_AgeRoll_CLAMS_081123.numbers"
)


# -------------------------------------------------
# Load spreadsheet
# -------------------------------------------------
file_ext = os.path.splitext(file_path)[1].lower()
all_sheets = {}

if file_ext == ".xlsx":
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

elif file_ext == ".xls":
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine="xlrd")

elif file_ext == ".numbers":

    from numbers_parser import Document

    doc = Document(file_path)

    for sheet in doc.sheets:

        if not sheet.tables:
            continue

        table = sheet.tables[0]

        data = []
        for row in table.rows():
            data.append([getattr(cell, "value", None) for cell in row])

        if data:
            all_sheets[sheet.name] = pd.DataFrame(data)

else:
    raise ValueError("Unsupported spreadsheet type")


sheet_names = list(all_sheets.keys())

meta_sheet = next((s for s in sheet_names if "metabolic" in s.lower()), None)
mouse_sheets = sorted([s for s in sheet_names if s.strip().isdigit()])

print("Metadata sheet:", meta_sheet)
print("Mouse sheets detected:", len(mouse_sheets))


# -------------------------------------------------
# Metadata
# -------------------------------------------------
meta_raw = all_sheets[meta_sheet]

meta = meta_raw.iloc[4:].copy()

meta = meta[meta[0].notna()].copy()

meta = meta.rename(columns={0: "mouse_label", 5: "age_months"})

meta["mouse_label"] = pd.to_numeric(meta["mouse_label"], errors="coerce").astype("Int64").astype(str)

meta["age_months"] = pd.to_numeric(meta["age_months"], errors="coerce")

meta = meta[["mouse_label", "age_months"]]


# -------------------------------------------------
# Load numeric sheets
# -------------------------------------------------
def load_numeric_sheet(sheet_name):

    df = all_sheets.get(sheet_name)

    if df is None or df.empty:
        return None

    df = df.applymap(lambda x: getattr(x, "value", x) if hasattr(x, "value") else x)

    if df.shape[1] > 1:

        df["segment"] = (
            df.iloc[:, 1]
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        df["segment"] = "UNKNOWN"

    numeric_cols = [c for c in df.columns if c != "segment"]

    df[numeric_cols] = (
        df[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
    )

    df = df.ffill().fillna(0)

    return df


# -------------------------------------------------
# Wavelet feature extraction
# -------------------------------------------------
def extract_wavelet_features(ts, wavelet="db4", level=3):

    coeffs = pywt.wavedec(ts, wavelet, level=level)

    features = []

    for c in coeffs:
        features.append(np.mean(c))
        features.append(np.std(c))
        features.append(np.max(c))
        features.append(np.min(c))

    return np.array(features)


# -------------------------------------------------
# Build feature matrices
# -------------------------------------------------
matrix_features = []
pca_features = []
wavelet_features = []

variable_matrices = {}
valid_mouse_labels = []

for sheet in mouse_sheets:

    df = load_numeric_sheet(sheet)

    if df is None:
        continue

    variable_matrices[sheet] = df

    numeric_cols = [c for c in df.columns if c != "segment"]

    data = df[numeric_cols].values.astype(float)

    data_scaled = StandardScaler().fit_transform(data)

    feature_vector = np.concatenate(
        [data_scaled.mean(axis=0), data_scaled.std(axis=0)]
    )

    n_components = min(5, data_scaled.shape[1], data_scaled.shape[0])

    pca_vec = PCA(n_components=n_components).fit_transform(data_scaled).flatten()

    ts = df[numeric_cols].mean(axis=1).astype(float).values

    wavelet_vec = extract_wavelet_features(ts)

    matrix_features.append(feature_vector)
    pca_features.append(pca_vec)
    wavelet_features.append(wavelet_vec)

    valid_mouse_labels.append(sheet)


if len(matrix_features) < 3:
    raise ValueError("Not enough mice for clustering")


# -------------------------------------------------
# Align feature lengths
# -------------------------------------------------
min_len1 = min(len(f) for f in matrix_features)
min_len2 = min(len(f) for f in pca_features)
min_len3 = min(len(f) for f in wavelet_features)

view1 = StandardScaler().fit_transform(
    np.array([f[:min_len1] for f in matrix_features])
)

view2 = StandardScaler().fit_transform(
    np.array([f[:min_len2] for f in pca_features])
)

view3 = StandardScaler().fit_transform(
    np.array([f[:min_len3] for f in wavelet_features])
)

X_scaled = view1.copy()


# -------------------------------------------------
# Automatic cluster selection
# -------------------------------------------------
max_k = min(8, len(X_scaled) - 1)
k_range = range(2, max_k + 1)


def select_best_k(X):

    sil_scores = []
    db_scores = []
    ch_scores = []

    for k in k_range:

        km = KMeans(n_clusters=k, random_state=42, n_init=20)

        labels = km.fit_predict(X)

        sil_scores.append(silhouette_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))

    sil_norm = (np.array(sil_scores) - np.min(sil_scores)) / (np.ptp(sil_scores) + 1e-9)
    db_norm = (np.max(db_scores) - np.array(db_scores)) / (np.ptp(db_scores) + 1e-9)
    ch_norm = (np.array(ch_scores) - np.min(ch_scores)) / (np.ptp(ch_scores) + 1e-9)

    combined = sil_norm + db_norm + ch_norm

    return list(k_range)[np.argmax(combined)]


best_k = select_best_k(X_scaled)

print("Optimal clusters:", best_k)


# -------------------------------------------------
# Standard clustering
# -------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)

cluster_labels = kmeans.fit_predict(X_scaled)

cluster_df = pd.DataFrame(
    {"mouse_label": valid_mouse_labels, "cluster": cluster_labels}
).merge(meta, on="mouse_label", how="left")


# -------------------------------------------------
# Multiview clustering
# -------------------------------------------------
mv_kmeans = MultiviewKMeans(n_clusters=best_k, random_state=42)

cluster_labels_mv = mv_kmeans.fit_predict([view1, view2, view3])

cluster_df["mv_cluster"] = cluster_labels_mv


# -------------------------------------------------
# PCA visualization
# -------------------------------------------------
X_pca = PCA(n_components=2).fit_transform(view1)

plt.figure(figsize=(8,6))

for c in np.unique(cluster_labels_mv):

    idx = cluster_labels_mv == c

    plt.scatter(X_pca[idx,0], X_pca[idx,1], label=f"Cluster {c}")

for i,label in enumerate(valid_mouse_labels):
    plt.text(X_pca[i,0], X_pca[i,1], label, fontsize=8)

plt.title("Multiview clustering PCA")
plt.legend()
plt.show()


# -------------------------------------------------
# Time series clustering
# -------------------------------------------------
ts_all = []
ts_labels = []

for sheet in valid_mouse_labels:

    df = variable_matrices[sheet]

    numeric_cols = [c for c in df.columns if c != "segment"]

    ts = df[numeric_cols].mean(axis=1).astype(float).values

    ts_all.append(ts)
    ts_labels.append(sheet)


max_len = max(len(ts) for ts in ts_all)

ts_all_padded = np.array([
    np.pad(ts.astype(float),(0,max_len-len(ts)),mode="constant")
    for ts in ts_all
])

ts_all_scaled = TimeSeriesScalerMeanVariance().fit_transform(
    to_time_series_dataset(ts_all_padded)
)

linkage_matrix = linkage(
    ts_all_scaled.reshape(len(ts_all_scaled), -1),
    method="ward"
)

plt.figure(figsize=(12,6))
dendrogram(linkage_matrix, labels=ts_labels)
plt.title("Hierarchical Time Series Clustering")
plt.show()

block_cluster_labels = fcluster(linkage_matrix, t=best_k, criterion="maxclust")

cluster_df["block_cluster"] = block_cluster_labels


# -------------------------------------------------
# Age boxplots
# -------------------------------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
cluster_df.boxplot(column="age_months", by="cluster")

plt.subplot(1,2,2)
cluster_df.boxplot(column="age_months", by="mv_cluster")

plt.show()


# -------------------------------------------------
# Segment plots by age groups
# -------------------------------------------------
age_groups = {
    "6-7": (6,7),
    "12-13": (12,13),
    "18-19": (18,19),
    "26-27": (26,27),
}

mouse_to_age_group = {}

for _,row in meta.iterrows():

    mouse=row["mouse_label"]
    age=row["age_months"]

    if pd.isna(age):
        continue

    for g,(low,high) in age_groups.items():

        if low<=age<=high:
            mouse_to_age_group[mouse]=g


numeric_cols_all = set.intersection(
    *[set([c for c in df.columns if c!="segment"]) for df in variable_matrices.values()]
)

var_names=sorted(list(numeric_cols_all))

all_segments=sorted(
    list(set(seg for df in variable_matrices.values() for seg in df["segment"].unique()))
)


for seg in all_segments:

    for var_name in var_names:

        plt.figure(figsize=(10,5))

        for group_name in age_groups.keys():

            mice_in_group=[
                m for m in valid_mouse_labels
                if mouse_to_age_group.get(m)==group_name
            ]

            ts_group=[]

            for m in mice_in_group:

                df=variable_matrices[m]

                df_seg=df[df["segment"].str.upper()==seg.upper()]

                if df_seg.empty:
                    continue

                ts=pd.to_numeric(df_seg[var_name],errors="coerce").dropna().astype(float).values

                if len(ts)>0:
                    ts_group.append(ts)

            if ts_group:

                max_len=max(len(ts) for ts in ts_group)

                ts_padded=np.array([
                    np.pad(ts.astype(float),(0,max_len-len(ts)),
                    mode="constant",constant_values=np.nan)
                    for ts in ts_group
                ])

                mean_ts=np.nanmean(ts_padded,axis=0)
                std_ts=np.nanstd(ts_padded,axis=0)

                plt.plot(mean_ts,label=f"Age {group_name}")

                plt.fill_between(
                    range(len(mean_ts)),
                    mean_ts-std_ts,
                    mean_ts+std_ts,
                    alpha=0.2
                )

        plt.title(f"{var_name} | Segment {seg}")
        plt.xlabel("Timepoint")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
        
        scales = np.arange(1,128)
coef, freq = pywt.cwt(ts, scales, 'morl')

plt.imshow(coef, aspect='auto', cmap='jet')
plt.xlabel("Time")
plt.ylabel("Scale (frequency)")
plt.title("Wavelet Spectrogram")
plt.colorbar()
plt.show()


# -------------------------------------------------
# Simplified statistical metrics per age group
# -------------------------------------------------

stats_rows = []

for group_name in age_groups.keys():

    mice_in_group = [
        m for m in valid_mouse_labels
        if mouse_to_age_group.get(m) == group_name
    ]

    for var in var_names:

        values = []

        for m in mice_in_group:

            df = variable_matrices[m]

            vals = pd.to_numeric(df[var], errors="coerce").dropna().values

            if len(vals) > 0:
                values.extend(vals)

        values = np.array(values)

        if len(values) == 0:
            continue

        # Shannon entropy
        hist, _ = np.histogram(values, bins=40, density=True)
        hist = hist + 1e-12
        shannon_entropy = entropy(hist)

        stats_rows.append({
            "Age Group": group_name,
            "Variable": var,
            "Variance": np.var(values),
            "Kurtosis": kurtosis(values),
            "Entropy": shannon_entropy
        })


stats_df = pd.DataFrame(stats_rows)



# -------------------------------------------------
# Plot table per age group
# -------------------------------------------------

stats_display = stats_df.copy()

numeric_cols = stats_display.select_dtypes(include=[np.number]).columns
stats_display[numeric_cols] = stats_display[numeric_cols].round(4)

for group in stats_display["Age Group"].unique():

    df_group = stats_display[stats_display["Age Group"] == group]

    fig, ax = plt.subplots(figsize=(10, len(df_group)*0.4 + 2))

    ax.axis("off")

    table = ax.table(
        cellText=df_group.values,
        colLabels=df_group.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1,1.4)

    plt.title(f"Variance, Kurtosis, and Entropy — Age Group {group}", fontsize=14)

    plt.tight_layout()
    plt.show()
    
    
    
    
    
    # -------------------------------------------------
# Build heatmap matrices
# -------------------------------------------------

variance_matrix = stats_df.pivot(
    index="Variable",
    columns="Age Group",
    values="Variance"
)

entropy_matrix = stats_df.pivot(
    index="Variable",
    columns="Age Group",
    values="Entropy"
)



# -------------------------------------------------
# Variance heatmap
# -------------------------------------------------

plt.figure(figsize=(8,6))

plt.imshow(variance_matrix, aspect="auto")

plt.colorbar(label="Variance")

plt.xticks(
    range(len(variance_matrix.columns)),
    variance_matrix.columns
)

plt.yticks(
    range(len(variance_matrix.index)),
    variance_matrix.index
)

plt.title("Metabolic Variance Across Age Groups")

plt.xlabel("Age Group (months)")
plt.ylabel("Variable")

plt.tight_layout()
plt.show()



# -------------------------------------------------
# Entropy heatmap
# -------------------------------------------------

plt.figure(figsize=(8,6))

plt.imshow(entropy_matrix, aspect="auto")

plt.colorbar(label="Shannon Entropy")

plt.xticks(
    range(len(entropy_matrix.columns)),
    entropy_matrix.columns
)

plt.yticks(
    range(len(entropy_matrix.index)),
    entropy_matrix.index
)

plt.title("Metabolic Signal Entropy Across Age Groups")

plt.xlabel("Age Group (months)")
plt.ylabel("Variable")

plt.tight_layout()
plt.show()



# -------------------------------------------------
# Circadian rhythm matrix (fixed)
# -------------------------------------------------

circadian_data = {}

for group_name in age_groups.keys():

    mice_in_group = [
        m for m in valid_mouse_labels
        if mouse_to_age_group.get(m) == group_name
    ]

    hourly_values = {var: [[] for _ in range(24)] for var in var_names}

    for m in mice_in_group:

        df = variable_matrices[m]

        for var in var_names:

            if var not in df.columns:
                continue

            ts = pd.to_numeric(df[var], errors="coerce").values

            for t, val in enumerate(ts):

                if np.isnan(val):
                    continue

                hour = t % 24

                hourly_values[var][hour].append(val)

    circadian_matrix = []

    for var in var_names:

        row = []

        for h in range(24):

            vals = hourly_values[var][h]

            if len(vals) > 0:
                row.append(np.mean(vals))
            else:
                row.append(np.nan)

        circadian_matrix.append(row)

    circadian_data[group_name] = np.array(circadian_matrix)    
    
    # -------------------------------------------------
# Circadian rhythm heatmaps
# -------------------------------------------------

for group_name, matrix in circadian_data.items():

    plt.figure(figsize=(10,6))

    plt.imshow(matrix, aspect="auto")

    plt.colorbar(label="Mean Value")

    plt.xticks(range(24), range(24))
    plt.yticks(range(len(var_names)), var_names)

    plt.xlabel("Hour of Day")
    plt.ylabel("Variable")

    plt.title(f"Circadian Metabolic Rhythm — Age {group_name}")

    plt.tight_layout()
    plt.show()
    
    # ==========================================================
# VAR2 vs VAR3 | SEPARATE PLOTS PER AGE GROUP AND SEGMENT
# ==========================================================

if len(var_names) < 3:
    raise ValueError("Not enough variables")

var2_name = var_names[5]
var3_name = var_names[4]

print(f"Plotting: {var2_name} and {var3_name}")

target_segments = ["OFF1", "ON1"]

for group_name in age_groups.keys():

    mice_in_group = [
        m for m in valid_mouse_labels
        if mouse_to_age_group.get(m) == group_name
    ]

    for seg in target_segments:

        ts_group_var2 = []
        ts_group_var3 = []

        for m in mice_in_group:

            df = variable_matrices[m]

            df_seg = df[df["segment"].str.upper() == seg.upper()]

            if df_seg.empty:
                continue

            v2 = pd.to_numeric(df_seg[var2_name], errors="coerce").dropna().values
            v3 = pd.to_numeric(df_seg[var3_name], errors="coerce").dropna().values

            if len(v2) > 0:
                ts_group_var2.append(v2)

            if len(v3) > 0:
                ts_group_var3.append(v3)

        # Skip empty cases
        if not ts_group_var2 and not ts_group_var3:
            continue

        plt.figure(figsize=(10,5))

        # -------------------------
        # VAR2
        # -------------------------
        if ts_group_var2:

            max_len = max(len(ts) for ts in ts_group_var2)

            ts_padded = np.array([
                np.pad(ts, (0, max_len - len(ts)),
                       mode="constant", constant_values=np.nan)
                for ts in ts_group_var2
            ])

            mean_ts = np.nanmean(ts_padded, axis=0)
            std_ts = np.nanstd(ts_padded, axis=0)

            plt.plot(mean_ts, linestyle='-', label=var2_name)
            plt.fill_between(
                range(len(mean_ts)),
                mean_ts - std_ts,
                mean_ts + std_ts,
                alpha=0.2
            )

        # -------------------------
        # VAR3
        # -------------------------
        if ts_group_var3:

            max_len = max(len(ts) for ts in ts_group_var3)

            ts_padded = np.array([
                np.pad(ts, (0, max_len - len(ts)),
                       mode="constant", constant_values=np.nan)
                for ts in ts_group_var3
            ])

            mean_ts = np.nanmean(ts_padded, axis=0)
            std_ts = np.nanstd(ts_padded, axis=0)

            plt.plot(mean_ts, linestyle='--', label=var3_name)
            plt.fill_between(
                range(len(mean_ts)),
                mean_ts - std_ts,
                mean_ts + std_ts,
                alpha=0.2
            )

        # -------------------------
        # Formatting
        # -------------------------
        plt.title(f"Age {group_name} | Segment {seg}")
        plt.xlabel("Timepoint")
        plt.ylabel("Signal")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
        
        
        
        from scipy.stats import pearsonr
import numpy as np
import pandas as pd

# -------------------------------------------------
# Select variable 2 (same logic as your code)
# -------------------------------------------------
var2_name = var_names[5]   # adjust index if needed

# -------------------------------------------------
# Get Mouse #1 OFF1 reference signal
# -------------------------------------------------
ref_mouse = "1"

df_ref = variable_matrices.get(ref_mouse)

if df_ref is None:
    raise ValueError("Mouse #1 not found")

df_ref_off1 = df_ref[df_ref["segment"].str.upper() == "OFF1"]

ref_ts = pd.to_numeric(df_ref_off1[var2_name], errors="coerce").dropna().values

if len(ref_ts) == 0:
    raise ValueError("Mouse #1 OFF1 segment is empty")

# -------------------------------------------------
# Compare with all other mice
# -------------------------------------------------
results = []

for mouse, df in variable_matrices.items():

    if mouse == ref_mouse:
        continue

    df_off1 = df[df["segment"].str.upper() == "OFF1"]

    ts = pd.to_numeric(df_off1[var2_name], errors="coerce").dropna().values

    if len(ts) == 0:
        continue

    # -------------------------------------------------
    # Align lengths (truncate to shortest)
    # -------------------------------------------------
    min_len = min(len(ref_ts), len(ts))

    ref_aligned = ref_ts[:min_len]
    ts_aligned = ts[:min_len]

    # -------------------------------------------------
    # Pearson correlation
    # -------------------------------------------------
    corr, _ = pearsonr(ref_aligned, ts_aligned)

    results.append({
        "mouse": mouse,
        "pearson_corr": corr,
        "length_used": min_len
    })

# -------------------------------------------------
# Sort by similarity (highest correlation first)
# -------------------------------------------------
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="pearson_corr", ascending=False)

print("\nMost similar OFF1 (Variable 2) to Mouse #1:\n")
print(results_df.head(10))
