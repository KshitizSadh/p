# ============================================================
# PRACTICAL 6: Logistic Regression
# Dataset: Breast Cancer Wisconsin (sklearn)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# -------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)          # 0=malignant, 1=benign

print("=" * 60)
print("PRACTICAL 6: Logistic Regression")
print("=" * 60)
print(f"\nDataset     : Breast Cancer Wisconsin")
print(f"Classes     : {list(data.target_names)} (0=malignant, 1=benign)")
print(f"Samples     : {len(X)}")
print(f"Class Dist  : {y.value_counts().to_dict()}")

# -------------------------------------------------------
# 2. Scale + Train-Test Split
# -------------------------------------------------------
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# -------------------------------------------------------
# 3. Train Model
# -------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------------------------------
# 4. Evaluation Metrics
# -------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

accuracy    = accuracy_score(y_test, y_pred)
error_rate  = 1 - accuracy
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)          # Sensitivity / TPR
specificity = TN / (TN + FP)
f1          = 2 * precision * recall / (precision + recall)
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

print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# -------------------------------------------------------
# 5. K-Fold Cross-Validation (k=10)
# -------------------------------------------------------
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_acc = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')
cv_auc = cross_val_score(model, X_scaled, y, cv=kf, scoring='roc_auc')

print("--- 10-Fold Cross-Validation ---")
print(f"CV Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"CV AUC-ROC   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# -------------------------------------------------------
# 6. Plots
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Practical 6 – Logistic Regression", fontsize=14, fontweight='bold')

# (a) Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names).plot(
    ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix")

# (b) ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={auc:.4f})')
axes[1].plot([0,1],[0,1], 'k--', lw=1)
axes[1].fill_between(fpr, tpr, alpha=0.15, color='darkorange')
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

# (c) CV Accuracy bars
axes[2].bar(range(1, 11), cv_acc, color='steelblue', edgecolor='navy')
axes[2].axhline(cv_acc.mean(), color='red', linestyle='--', label=f'Mean={cv_acc.mean():.3f}')
axes[2].set_xlabel("Fold")
axes[2].set_ylabel("Accuracy")
axes[2].set_title("10-Fold CV Accuracy")
axes[2].legend()
axes[2].set_ylim([0.8, 1.0])

plt.tight_layout()
plt.savefig("Practical_06_Logistic_Regression.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_06_Logistic_Regression.png")
