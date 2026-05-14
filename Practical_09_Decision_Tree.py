# ============================================================
# PRACTICAL 9: Decision Tree Classification
# Dataset: Breast Cancer Wisconsin (sklearn)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# -------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("=" * 60)
print("PRACTICAL 9: Decision Tree Classification")
print("=" * 60)
print(f"\nDataset : Breast Cancer Wisconsin")
print(f"Classes : {list(data.target_names)}")
print(f"Samples : {len(X)}")

# -------------------------------------------------------
# 2. Train-Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# -------------------------------------------------------
# 3. Hyperparameter Tuning: max_depth
# -------------------------------------------------------
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
depth_range = range(1, 15)
cv_acc_list = []

print("\n--- Max Depth vs CV Accuracy ---")
for d in depth_range:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42, criterion='gini')
    score = cross_val_score(dt, X, y, cv=kf, scoring='accuracy').mean()
    cv_acc_list.append(score)

best_depth = depth_range[np.argmax(cv_acc_list)]
print(f"Best max_depth: {best_depth}  (CV Accuracy={max(cv_acc_list):.4f})")

# -------------------------------------------------------
# 4. Train Final Model
# -------------------------------------------------------
model = DecisionTreeClassifier(
    max_depth=best_depth,
    criterion='gini',
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------------------------------
# 5. Evaluation Metrics
# -------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

accuracy    = accuracy_score(y_test, y_pred)
error_rate  = 1 - accuracy
precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
recall      = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
f1          = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0)
auc         = roc_auc_score(y_test, y_prob)

print("\n--- Confusion Matrix ---")
print(f"  TN={TN}  FP={FP}")
print(f"  FN={FN}  TP={TP}")
print("\n--- Classification Metrics ---")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Error Rate   : {error_rate:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall (TPR) : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"AUC-ROC      : {auc:.4f}")
print(f"\nTree Depth   : {model.get_depth()}")
print(f"Leaf Nodes   : {model.get_n_leaves()}")

print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# -------------------------------------------------------
# 6. K-Fold Cross-Validation (k=10)
# -------------------------------------------------------
cv_acc = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
cv_auc = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
print("--- 10-Fold Cross-Validation ---")
print(f"CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"CV AUC-ROC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# -------------------------------------------------------
# 7. Feature Importances
# -------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=data.feature_names)
top10 = importances.sort_values(ascending=False).head(10)
print("\n--- Top 10 Feature Importances ---")
print(top10.round(4).to_string())

# -------------------------------------------------------
# 8. Plots
# -------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Practical 9 – Decision Tree (max_depth={best_depth})",
             fontsize=14, fontweight='bold')

# (a) Depth vs CV Accuracy
axes[0,0].plot(list(depth_range), cv_acc_list, 'o-', color='steelblue', lw=2)
axes[0,0].axvline(best_depth, color='red', linestyle='--', label=f'Best depth={best_depth}')
axes[0,0].set_xlabel("Max Depth")
axes[0,0].set_ylabel("CV Accuracy")
axes[0,0].set_title("Max Depth vs CV Accuracy")
axes[0,0].legend()

# (b) Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names).plot(
    ax=axes[0,1], colorbar=False, cmap='Oranges')
axes[0,1].set_title("Confusion Matrix")

# (c) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={auc:.4f}')
axes[1,0].plot([0,1],[0,1],'k--',lw=1)
axes[1,0].fill_between(fpr, tpr, alpha=0.15, color='darkorange')
axes[1,0].set_xlabel("FPR")
axes[1,0].set_ylabel("TPR")
axes[1,0].set_title("ROC Curve")
axes[1,0].legend()

# (d) Feature Importances (Top 10)
top10.sort_values().plot(kind='barh', ax=axes[1,1], color='teal')
axes[1,1].set_title("Top 10 Feature Importances")
axes[1,1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("Practical_09_Decision_Tree.png", dpi=150, bbox_inches='tight')
plt.show()

# Separate tree visualization
fig2, ax2 = plt.subplots(figsize=(20, 10))
plot_tree(model, feature_names=data.feature_names,
          class_names=data.target_names, filled=True,
          rounded=True, fontsize=8, ax=ax2, max_depth=3)
ax2.set_title(f"Decision Tree (showing top 3 levels)", fontsize=14)
plt.tight_layout()
plt.savefig("Practical_09_Decision_Tree_Structure.png", dpi=120, bbox_inches='tight')
plt.show()
print("\nPlots saved as Practical_09_Decision_Tree.png and Practical_09_Decision_Tree_Structure.png")
