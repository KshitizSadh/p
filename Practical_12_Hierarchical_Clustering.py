# ============================================================
# PRACTICAL 12: Hierarchical Clustering (Agglomerative)
# Dataset: Customer Segmentation (same as Practical 11)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# -------------------------------------------------------
# 1. Generate / Load Dataset
# -------------------------------------------------------
try:
    df = pd.read_csv("Mall_Customers.csv")
    df.columns = df.columns.str.strip()
    X_raw = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    print("Loaded Mall_Customers.csv")
except FileNotFoundError:
    np.random.seed(42)
    X_raw = np.vstack([
        np.random.multivariate_normal([25, 30, 80], [[25,0,0],[0,50,0],[0,0,80]], 40),
        np.random.multivariate_normal([45, 80, 20], [[50,0,0],[0,80,0],[0,0,50]], 40),
        np.random.multivariate_normal([30, 80, 80], [[30,0,0],[0,60,0],[0,0,70]], 40),
        np.random.multivariate_normal([50, 30, 30], [[40,0,0],[0,50,0],[0,0,60]], 40),
        np.random.multivariate_normal([35, 55, 55], [[35,0,0],[0,55,0],[0,0,65]], 40),
    ])
    X_raw = np.clip(X_raw, [15, 10, 1], [70, 140, 100])
    print("Generated synthetic customer dataset (200 samples)")

print("=" * 60)
print("PRACTICAL 12: Hierarchical Clustering (Agglomerative)")
print("=" * 60)
print(f"\nDataset : Customer Segmentation")
print(f"Features: Age, Annual Income (k$), Spending Score")
print(f"Samples : {len(X_raw)}")

# -------------------------------------------------------
# 2. Scale
# -------------------------------------------------------
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# -------------------------------------------------------
# 3. Linkage Methods Comparison
# -------------------------------------------------------
linkage_methods = ['ward', 'complete', 'average', 'single']
n_clusters_test = 5

print("\n--- Linkage Method Comparison (K=5) ---")
print(f"{'Method':>10} | {'Silhouette':>11} | {'Davies-Bouldin':>15} | {'Calinski-H':>11}")
print("-" * 57)
link_results = {}
for method in linkage_methods:
    if method == 'ward':
        agg = AgglomerativeClustering(n_clusters=n_clusters_test, linkage='ward')
    else:
        agg = AgglomerativeClustering(n_clusters=n_clusters_test, linkage=method)
    labels = agg.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    cal = calinski_harabasz_score(X_scaled, labels)
    link_results[method] = {'sil': sil, 'db': db, 'cal': cal, 'labels': labels}
    print(f"{method:>10} | {sil:>11.4f} | {db:>15.4f} | {cal:>11.2f}")

best_method = max(link_results, key=lambda m: link_results[m]['sil'])
print(f"\nBest Linkage Method (by Silhouette): {best_method}")

# -------------------------------------------------------
# 4. Find Optimal Number of Clusters (Ward linkage)
# -------------------------------------------------------
k_range     = range(2, 10)
silhouettes = []
db_scores   = []

print("\n--- Silhouette Score vs Number of Clusters (Ward) ---")
for k in k_range:
    agg    = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels)
    db     = davies_bouldin_score(X_scaled, labels)
    silhouettes.append(sil)
    db_scores.append(db)
    print(f"  K={k}: Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}")

best_k = list(k_range)[np.argmax(silhouettes)]
print(f"\nBest K (Ward, Silhouette): {best_k}")

# -------------------------------------------------------
# 5. Final Model
# -------------------------------------------------------
final_model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
labels      = final_model.fit_predict(X_scaled)

sil  = silhouette_score(X_scaled, labels)
db   = davies_bouldin_score(X_scaled, labels)
cal  = calinski_harabasz_score(X_scaled, labels)

print(f"\n--- Final Model: Ward Linkage, K={best_k} ---")
print(f"Silhouette Score     : {sil:.4f}  (↑ better)")
print(f"Davies-Bouldin Score : {db:.4f}   (↓ better)")
print(f"Calinski-Harabasz    : {cal:.2f}  (↑ better)")

print("\n--- Cluster Sizes ---")
unique, counts = np.unique(labels, return_counts=True)
for cl, cnt in zip(unique, counts):
    print(f"  Cluster {cl}: {cnt} samples")

print("\n--- Cluster Centres (Original Scale) ---")
for i in range(best_k):
    mask   = labels == i
    centre = X_raw[mask].mean(axis=0)
    print(f"  Cluster {i}: Age={centre[0]:.1f}  Income={centre[1]:.1f}  Spending={centre[2]:.1f}")

# -------------------------------------------------------
# 6. Compute Linkage Matrix for Dendrogram
# -------------------------------------------------------
# Use a subset for dendrogram clarity (max 80 samples)
n_dend = min(80, len(X_scaled))
idx    = np.random.choice(len(X_scaled), n_dend, replace=False)
Z      = linkage(X_scaled[idx], method='ward')

# -------------------------------------------------------
# 7. PCA for 2D Visualization
# -------------------------------------------------------
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA Variance Explained: {pca.explained_variance_ratio_.sum():.4f}")

# -------------------------------------------------------
# 8. Plots
# -------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 0.9, best_k))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Practical 12 – Hierarchical Clustering (Ward, K={best_k})",
             fontsize=14, fontweight='bold')

# (a) Dendrogram
dendrogram(Z, ax=axes[0,0], truncate_mode='lastp', p=25,
           color_threshold=0.7 * max(Z[:,2]),
           leaf_font_size=7, above_threshold_color='gray')
axes[0,0].set_title(f"Dendrogram (Ward Linkage, n={n_dend} samples)")
axes[0,0].set_xlabel("Sample Index")
axes[0,0].set_ylabel("Distance")

# (b) Silhouette vs K
axes[0,1].plot(list(k_range), silhouettes, 'o-', color='darkorange', lw=2)
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
axes[1,0].set_xlabel("PC1")
axes[1,0].set_ylabel("PC2")
axes[1,0].set_title("Clusters (PCA 2D Projection)")
axes[1,0].legend(fontsize=8)

# (d) Income vs Spending Scatter (original features)
for i in range(best_k):
    mask = labels == i
    axes[1,1].scatter(X_raw[mask, 1], X_raw[mask, 2],
                      color=colors[i], s=30, alpha=0.7, label=f'Cluster {i}')
axes[1,1].set_xlabel("Annual Income (k$)")
axes[1,1].set_ylabel("Spending Score (1-100)")
axes[1,1].set_title("Income vs Spending by Cluster")
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("Practical_12_Hierarchical_Clustering.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_12_Hierarchical_Clustering.png")

# -------------------------------------------------------
# 9. Linkage Method Comparison Bar Chart
# -------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 4))
method_names = list(link_results.keys())
sil_scores   = [link_results[m]['sil'] for m in method_names]
bars = ax2.bar(method_names, sil_scores,
               color=['green' if m == best_method else 'steelblue' for m in method_names],
               edgecolor='black')
ax2.set_ylabel("Silhouette Score")
ax2.set_title(f"Linkage Method Comparison (K={n_clusters_test})")
for bar, val in zip(bars, sil_scores):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("Practical_12_Linkage_Comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as Practical_12_Linkage_Comparison.png")
