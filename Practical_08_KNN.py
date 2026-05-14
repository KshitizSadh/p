# ============================================================
# PRACTICAL 8: K-Nearest Neighbours (K-NN) Classifier
# Dataset: Iris Dataset (UCI / sklearn)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize

# -------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

print("=" * 60)
print("PRACTICAL 8: K-Nearest Neighbours (K-NN) Classifier")
print("=" * 60)
print(f"\nDataset : Iris")
print(f"Classes : {list(iris.target_names)}")
print(f"Samples : {len(X)}")

# -------------------------------------------------------
# 2. Scale + Train-Test Split
# -------------------------------------------------------
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# -------------------------------------------------------
# 3. Find Optimal K (Elbow Method)
# -------------------------------------------------------
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
k_range   = range(1, 31)
cv_scores = []

print("\n--- K vs CV Accuracy ---")
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy').mean()
    cv_scores.append(score)

best_k = k_range[np.argmax(cv_scores)]
print(f"Best K (by 10-fold CV): {best_k}  (Accuracy={max(cv_scores):.4f})")

# -------------------------------------------------------
# 4. Train Final Model with Best K
# -------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# -------------------------------------------------------
# 5. Evaluation Metrics
# -------------------------------------------------------
cm       = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Confusion Matrix ---")
print(cm)
print("\n--- Per-Class TP / TN / FP / FN / Specificity ---")
for i, cls in enumerate(iris.target_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - TP - FP - FN
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    print(f"  {cls:12s} | TP={TP}  TN={TN}  FP={FP}  FN={FN}  Spec={spec:.4f}")

print(f"\nAccuracy   : {accuracy:.4f}")
print(f"Error Rate : {1 - accuracy:.4f}")

report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
print(f"Precision  : {report['macro avg']['precision']:.4f}")
print(f"Recall     : {report['macro avg']['recall']:.4f}")
print(f"F1-Score   : {report['macro avg']['f1-score']:.4f}")

y_test_bin = label_binarize(y_test, classes=[0,1,2])
auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
print(f"AUC (OvR)  : {auc:.4f}")

print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# -------------------------------------------------------
# 6. K-Fold CV with Best K
# -------------------------------------------------------
cv_acc = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')
print(f"--- 10-Fold CV (K={best_k}) ---")
print(f"Fold Scores : {np.round(cv_acc, 4)}")
print(f"Mean        : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

# -------------------------------------------------------
# 7. Decision Boundary (using first 2 features)
# -------------------------------------------------------
X2 = X_scaled[:, :2]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42, stratify=y
)
knn2 = KNeighborsClassifier(n_neighbors=best_k)
knn2.fit(X2_train, y2_train)

h = 0.02
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# -------------------------------------------------------
# 8. Plots
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Practical 8 – K-NN Classifier (Best K={best_k})",
             fontsize=14, fontweight='bold')

# (a) K vs CV Accuracy (Elbow)
axes[0].plot(list(k_range), cv_scores, 'o-', color='steelblue', lw=2)
axes[0].axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0].set_xlabel("Number of Neighbours (K)")
axes[0].set_ylabel("CV Accuracy")
axes[0].set_title("Elbow Curve: K vs Accuracy")
axes[0].legend()

# (b) Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot(
    ax=axes[1], colorbar=False, cmap='Greens')
axes[1].set_title("Confusion Matrix")

# (c) Decision Boundary (2D)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold  = ['#FF0000', '#00AA00', '#0000FF']
axes[2].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
for cls_idx, color in enumerate(cmap_bold):
    idx = y == cls_idx
    axes[2].scatter(X2[idx, 0], X2[idx, 1], c=color, s=20,
                    edgecolor='k', lw=0.3, label=iris.target_names[cls_idx])
axes[2].set_xlabel("Feature 1 (scaled)")
axes[2].set_ylabel("Feature 2 (scaled)")
axes[2].set_title(f"Decision Boundary (K={best_k})")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("Practical_08_KNN.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_08_KNN.png")
