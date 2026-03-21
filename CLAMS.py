# -*- coding: utf-8 -*-
"""
CLAMS clustering using matrix-based features
Automatic optimal cluster selection
Includes PCA visualization and multiview clustering
ROBUST VERSION: handles .xls, .xlsx, and .numbers
"""

# -------------------------------
# Imports
# -------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA
from mvlearn.cluster import MultiviewKMeans
from scipy.stats import f_oneway
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

os.environ["OMP_NUM_THREADS"] = "1"

# -------------------------------
# File path
# -------------------------------
file_path = os.path.join(
    os.path.dirname(__file__),
    "SimplifiedTerskikh10_AgeRoll_CLAMS_081123.numbers"
)

# -------------------------------
# Detect engine and load sheets
# -------------------------------
file_ext = os.path.splitext(file_path)[1].lower()
all_sheets = {}

if file_ext == ".xlsx":
    engine = "openpyxl"
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine=engine)
elif file_ext == ".xls":
    engine = "xlrd"
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine=engine)
elif file_ext == ".numbers":
    try:
        from numbers_parser import Document
    except ImportError:
        raise ImportError("Install 'numbers-parser': pip install numbers-parser")

    doc = Document(file_path)
    for sheet in doc.sheets:
        if not sheet.tables:
            continue
        table = sheet.tables[0]
        data = []
        for row in table.rows():
            row_values = [getattr(cell, "value", None) for cell in row]
            data.append(row_values)
        if data:
            all_sheets[sheet.name] = pd.DataFrame(data)
else:
    raise ValueError(f"Unsupported file extension: {file_ext}")

sheet_names = list(all_sheets.keys())
if not sheet_names:
    raise ValueError("No sheets detected in file.")

# Detect metadata sheet and mouse sheets
meta_sheet = next((s for s in sheet_names if "metabolic" in s.lower()), None)
mouse_sheets = sorted([s for s in sheet_names if s.strip().isdigit()])

if not meta_sheet:
    raise ValueError("No metadata sheet found.")
print(f"Metadata sheet: {meta_sheet}")
print(f"Detected {len(mouse_sheets)} mouse sheets")

# -------------------------------
# Load metadata
# -------------------------------
meta_raw = all_sheets[meta_sheet]
meta = meta_raw.iloc[4:].copy()  # Skip first 4 rows
meta = meta[meta[0].notna()].copy()
meta = meta.rename(columns={0: "mouse_label", 5: "age_months"})
meta["mouse_label"] = pd.to_numeric(meta["mouse_label"], errors="coerce").astype(int).astype(str)
meta["age_months"] = pd.to_numeric(meta["age_months"], errors="coerce")
meta = meta[["mouse_label", "age_months"]]

# -------------------------------
# Helper function to load numeric sheets
# -------------------------------
def load_numeric_sheet(sheet_name):
    """
    Returns a DataFrame with:
    - 'time': first column (for plotting)
    - numeric_cols: remaining columns converted to numeric (for clustering)
    """
    df = all_sheets.get(sheet_name)
    if df is None or df.empty:
        return None

    # Extract raw values
    df = df.applymap(lambda x: getattr(x, "value", x) if hasattr(x, "value") else x)
    
    # Preserve first column as time for plotting
    df['time'] = df.iloc[:, 0]

    # Convert remaining columns to numeric (exclude first column)
    numeric_cols = df.columns[1:-1]  # exclude 'time'
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    return df.ffill().fillna(0)

# ==========================================================
# BUILD FEATURE MATRICES (EXCLUDE TIME COLUMN)
# ==========================================================
matrix_features = []
pca_features = []
valid_mouse_labels = []

for sheet in mouse_sheets:
    df = load_numeric_sheet(sheet)
    if df is None:
        continue

    # Exclude 'time' column for clustering
    feature_data = df.drop(columns=['time']).values
    data_scaled = StandardScaler().fit_transform(feature_data)

    # View 1: mean + std
    feature_vector = np.concatenate([data_scaled.mean(axis=0), data_scaled.std(axis=0)])
    # View 2: PCA
    n_components = min(5, data_scaled.shape[1], data_scaled.shape[0])
    pca_vec = PCA(n_components=n_components).fit_transform(data_scaled).flatten()

    matrix_features.append(feature_vector)
    pca_features.append(pca_vec)
    valid_mouse_labels.append(sheet)

if len(matrix_features) < 3:
    raise ValueError("Not enough valid mouse sheets for clustering")

# -------------------------------
# Align feature lengths
# -------------------------------
min_len1 = min(len(f) for f in matrix_features)
min_len2 = min(len(f) for f in pca_features)
view1 = StandardScaler().fit_transform(np.array([f[:min_len1] for f in matrix_features]))
view2 = StandardScaler().fit_transform(np.array([f[:min_len2] for f in pca_features]))
X_scaled = view1.copy()

# ==========================================================
# AUTOMATIC CLUSTER SELECTION
# ==========================================================
max_k = min(8, len(X_scaled) - 1)
k_range = range(2, max_k + 1)

def select_best_k(X):
    sil_scores, db_scores, ch_scores = [], [], []
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
print(f"\nOptimal number of clusters: {best_k}")

# ==========================================================
# STANDARD KMEANS
# ==========================================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_scaled)
cluster_df = pd.DataFrame({"mouse_label": valid_mouse_labels, "cluster": cluster_labels}).merge(meta, on="mouse_label", how="left")
print("\nStandard clustering complete.")

# ==========================================================
# MULTIVIEW KMEANS
# ==========================================================
best_k_mv = select_best_k(view1)
mv_kmeans = MultiviewKMeans(n_clusters=best_k_mv, random_state=42, n_init=20)
cluster_labels_mv = mv_kmeans.fit_predict([view1, view2])
cluster_df["mv_cluster"] = cluster_labels_mv
print("Multiview clustering complete.")

# ==========================================================
# PCA Visualization (time excluded)
# ==========================================================
X_pca = PCA(n_components=2).fit_transform(view1)
plt.figure(figsize=(8, 6))
for c in np.unique(cluster_labels_mv):
    idx = cluster_labels_mv == c
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {c}")
for i, label in enumerate(valid_mouse_labels):
    plt.text(X_pca[i, 0], X_pca[i, 1], label, fontsize=8)
plt.title("Multiview Clustering PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# PLOT TIME SERIES VARIABLE BY CLUSTER (SUM, WITH TIME)
# ==========================================================
plt.figure(figsize=(12, 6))
for sheet in valid_mouse_labels:
    df = load_numeric_sheet(sheet)
    if df is None:
        continue
    ts = df.drop(columns=['time']).sum(axis=1).values
    cluster = cluster_df.loc[cluster_df["mouse_label"] == sheet, "mv_cluster"].values[0]
    plt.plot(df['time'], ts, color=f"C{cluster}", alpha=0.5)
for c in np.unique(cluster_df["mv_cluster"]):
    plt.plot([], [], color=f"C{c}", label=f"Cluster {c}")  # dummy lines for legend
plt.title("Food Time Series by Multiview Cluster")
plt.xlabel("Time")
plt.ylabel("Sum across features")
plt.legend()
plt.tight_layout()
plt.show()




    
    
    # ==========================================================
# AVERAGE PLOTS BY AGE GROUP (VARIABLE 2 & 3, SEGMENTS OFF1 & ON1)
# ==========================================================

plt.figure(figsize=(12, 6))

# Containers for aggregation
group_data = {}

for sheet in valid_mouse_labels:
    df = load_numeric_sheet(sheet)
    if df is None:
        continue

    # Get age
    age = cluster_df.loc[
        cluster_df["mouse_label"] == sheet, "age_months"
    ].values

    if len(age) == 0 or np.isnan(age[0]):
        continue

    age = int(age[0])

    # -------------------------
    # Extract variables
    # -------------------------
    # Adjust column indices if needed
    var2 = df.iloc[:, 2]  # variable 2
    var3 = df.iloc[:, 3]  # variable 3

    # -------------------------
    # Segment filtering
    # -------------------------
    # ASSUMPTION:
    # Column 1 contains segment labels like 'OFF1', 'ON1'
    segment_col = df.iloc[:, 1].astype(str)

    mask = segment_col.str.contains("OFF1|ON1", case=False, na=False)

    time = df.loc[mask, 'time']
    signal = (var2 + var3)[mask]  # combine variables

    if len(signal) == 0:
        continue

    # Normalize length (important for averaging)
    signal = signal.values

    if age not in group_data:
        group_data[age] = []

    group_data[age].append(signal)

# -------------------------
# Compute and plot averages
# -------------------------
for age, signals in group_data.items():
    # Make equal length
    min_len = min(len(s) for s in signals)
    aligned = np.array([s[:min_len] for s in signals])

    avg_signal = aligned.mean(axis=0)

    plt.plot(avg_signal, label=f"Age {age} months")

plt.title("Average Time Series (Var2 + Var3) for OFF1 & ON1 by Age")
plt.xlabel("Time Index (aligned)")
plt.ylabel("Average Signal")
plt.legend()
plt.tight_layout()
plt.show()