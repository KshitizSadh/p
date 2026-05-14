# ============================================================
# PRACTICAL 11: K-Means Clustering
# Dataset: Mall Customer Segmentation (UCI / Kaggle style)
#          (If unavailable online, synthetic version is generated)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# -------------------------------------------------------
# 1. Load / Generate Dataset
#    (Mall Customer Segmentation: Age, Income, Spending Score)
# -------------------------------------------------------
try:
    # Try to load if file exists locally
    df = pd.read_csv("Mall_Customers.csv")
    df.columns = df.columns.str.strip()
    X_raw = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    print("Loaded Mall_Customers.csv")
except FileNotFoundError:
    # Generate synthetic customer data
    np.random.seed(42)
    n = 200
    X_raw = np.vstack([
        np.random.multivariate_normal([25, 30, 80], [[25,0,0],[0,50,0],[0,0,80]], 40),   # young, low income, high spender
        np.random.multivariate_normal([45, 80, 20], [[50,0,0],[0,80,0],[0,0,50]], 40),   # mid age, high income, low spender
        np.random.multivariate_normal([30, 80, 80], [[30,0,0],[0,60,0],[0,0,70]], 40),   # young, high income, high spender
        np.random.multivariate_normal([50, 30, 30], [[40,0,0],[0,50,0],[0,0,60]], 40),   # older, low income, low spender
        np.random.multivariate_normal([35, 55, 55], [[35,0,0],[0,55,0],[0,0,65]], 40),   # average segment
    ])
    X_raw = np.clip(X_raw, [15, 10, 1], [70, 140, 100])
    df = pd.DataFrame(X_raw, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    print("Generated synthetic customer dataset (200 samples, 5 true clusters)")

print("=" * 60)
print("PRACTICAL 11: K-Means Clustering")
print("=" * 60)
print(f"\nDataset : Customer Segmentation")
print(f"Features: Age, Annual Income (k$), Spending Score")
print(f"Samples : {len(df)}")
print(f"\nData Statistics:")
print(df.describe().round(2).to_string())

# -------------------------------------------------------
# 2. Preprocess (Scale)
# -------------------------------------------------------
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# -------------------------------------------------------
# 3. Elbow Method + Silhouette to Find Optimal K
# -------------------------------------------------------
k_range  = range(2, 11)
inertias = []
silhouettes      = []
davies_bouldins  = []
calinski_scores  = []

print("\n--- Cluster Quality Metrics vs K ---")
print(f"{'K':>3} | {'Inertia':>10} | {'Silhouette':>11} | {'Davies-Bouldin':>15} | {'Calinski-H':>11}")
print("-" * 60)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    davies_bouldins.append(davies_bouldin_score(X_scaled, labels))
    calinski_scores.append(calinski_harabasz_score(X_scaled, labels))

    print(f"{k:>3} | {km.inertia_:>10.2f} | {silhouettes[-1]:>11.4f} | "
          f"{davies_bouldins[-1]:>15.4f} | {calinski_scores[-1]:>11.2f}")

# Choose best K by silhouette score
best_k = list(k_range)[np.argmax(silhouettes)]
print(f"\nBest K (by Silhouette Score): {best_k}")

# -------------------------------------------------------
# 4. Train Final K-Means Model
# -------------------------------------------------------
model  = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)
centers_scaled = model.cluster_centers_
centers_orig   = scaler.inverse_transform(centers_scaled)

print(f"\n--- K={best_k} Cluster Centres (Original Scale) ---")
centre_df = pd.DataFrame(centers_orig, columns=['Age', 'Annual Income (k$)', 'Spending Score'])
print(centre_df.round(2).to_string())

print(f"\n--- Cluster Sizes ---")
unique, counts = np.unique(labels, return_counts=True)
for cl, cnt in zip(unique, counts):
    print(f"  Cluster {cl}: {cnt} samples")

# Final metrics
sil  = silhouette_score(X_scaled, labels)
db   = davies_bouldin_score(X_scaled, labels)
cal  = calinski_harabasz_score(X_scaled, labels)
print(f"\nFinal Silhouette Score    : {sil:.4f}  (higher is better, range [-1,1])")
print(f"Final Davies-Bouldin Score: {db:.4f}  (lower is better)")
print(f"Final Calinski-Harabasz   : {cal:.2f}  (higher is better)")

# -------------------------------------------------------
# 5. PCA for 2D Visualization
# -------------------------------------------------------
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(centers_scaled)
print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_.round(4)}")
print(f"Total Variance Explained    : {pca.explained_variance_ratio_.sum():.4f}")

# -------------------------------------------------------
# 6. Plots
# -------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 0.9, best_k))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Practical 11 – K-Means Clustering (K={best_k})",
             fontsize=14, fontweight='bold')

# (a) Elbow Curve
axes[0,0].plot(list(k_range), inertias, 'o-', color='steelblue', lw=2)
axes[0,0].axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0,0].set_xlabel("Number of Clusters (K)")
axes[0,0].set_ylabel("Inertia (WCSS)")
axes[0,0].set_title("Elbow Method")
axes[0,0].legend()

# (b) Silhouette Score vs K
axes[0,1].plot(list(k_range), silhouettes, 's-', color='darkorange', lw=2)
axes[0,1].axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0,1].set_xlabel("Number of Clusters (K)")
axes[0,1].set_ylabel("Silhouette Score")
axes[0,1].set_title("Silhouette Score vs K")
axes[0,1].legend()

# (c) PCA 2D Cluster Plot
for i in range(best_k):
    mask = labels == i
    axes[1,0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                      color=colors[i], s=30, alpha=0.7, label=f'Cluster {i}')
axes[1,0].scatter(centers_pca[:,0], centers_pca[:,1],
                  c='black', s=200, marker='X', zorder=5, label='Centroids')
axes[1,0].set_xlabel("PC1")
axes[1,0].set_ylabel("PC2")
axes[1,0].set_title(f"Clusters (PCA 2D Projection)")
axes[1,0].legend(fontsize=8)

# (d) Income vs Spending Score (original features, coloured by cluster)
for i in range(best_k):
    mask = labels == i
    axes[1,1].scatter(X_raw[mask, 1], X_raw[mask, 2],
                      color=colors[i], s=30, alpha=0.7, label=f'Cluster {i}')
axes[1,1].scatter(centers_orig[:,1], centers_orig[:,2],
                  c='black', s=200, marker='X', zorder=5, label='Centroids')
axes[1,1].set_xlabel("Annual Income (k$)")
axes[1,1].set_ylabel("Spending Score (1-100)")
axes[1,1].set_title("Income vs Spending Score by Cluster")
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("Practical_11_KMeans.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_11_KMeans.png")
