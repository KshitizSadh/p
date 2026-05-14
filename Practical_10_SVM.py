# ============================================================
# PRACTICAL 10: Support Vector Machine (SVM) Classification
# Dataset: Breast Cancer Wisconsin (sklearn)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
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
y = pd.Series(data.target)

print("=" * 60)
print("PRACTICAL 10: SVM Classification")
print("=" * 60)
print(f"\nDataset : Breast Cancer Wisconsin")
print(f"Classes : {list(data.target_names)}")
print(f"Samples : {len(X)}")

# -------------------------------------------------------
# 2. Scale + Split
# -------------------------------------------------------
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# -------------------------------------------------------
# 3. Compare Kernels
# -------------------------------------------------------
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Kernel Comparison (5-Fold CV) ---")
print(f"{'Kernel':>10} | {'CV Acc Mean':>12} | {'CV Acc Std':>11}")
print("-" * 40)

kernel_results = {}
for k in kernels:
    svm = SVC(kernel=k, probability=True, random_state=42, C=1.0, gamma='scale')
    cv_acc = cross_val_score(svm, X_scaled, y, cv=kf, scoring='accuracy')
    kernel_results[k] = cv_acc
    print(f"{k:>10} | {cv_acc.mean():>12.4f} | {cv_acc.std():>11.4f}")

best_kernel = max(kernel_results, key=lambda k: kernel_results[k].mean())
print(f"\nBest Kernel: {best_kernel}")

# -------------------------------------------------------
# 4. Grid Search for Best C and Gamma (RBF)
# -------------------------------------------------------
print("\n--- Grid Search: C and Gamma (RBF kernel) ---")
param_grid = {'C': [0.01, 0.1, 1, 10, 100],
              'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]}
gs = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42),
                  param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)
print(f"Best Params : {gs.best_params_}")
print(f"Best CV Acc : {gs.best_score_:.4f}")

# -------------------------------------------------------
# 5. Train Final SVM with Best Params
# -------------------------------------------------------
best_params = gs.best_params_
model = SVC(kernel='rbf', probability=True, random_state=42, **best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------------------------------
# 6. Evaluation Metrics
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
print(f"Support Vecs : {model.n_support_} (malignant, benign)")

print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# -------------------------------------------------------
# 7. 10-Fold Cross-Validation
# -------------------------------------------------------
kf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_acc = cross_val_score(model, X_scaled, y, cv=kf10, scoring='accuracy')
cv_auc = cross_val_score(model, X_scaled, y, cv=kf10, scoring='roc_auc')
print("--- 10-Fold CV ---")
print(f"CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"CV AUC-ROC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# -------------------------------------------------------
# 8. Plots
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Practical 10 – SVM Classification (RBF Kernel)",
             fontsize=14, fontweight='bold')

# (a) Kernel comparison
kernel_means = [kernel_results[k].mean() for k in kernels]
kernel_stds  = [kernel_results[k].std()  for k in kernels]
bars = axes[0].bar(kernels, kernel_means, yerr=kernel_stds,
                   color=['steelblue','darkorange','green','red'],
                   capsize=5, edgecolor='black')
axes[0].set_ylabel("CV Accuracy")
axes[0].set_title("Kernel Comparison")
axes[0].set_ylim([0.8, 1.0])
for bar, val in zip(bars, kernel_means):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# (b) Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names).plot(
    ax=axes[1], colorbar=False, cmap='Reds')
axes[1].set_title("Confusion Matrix")

# (c) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'SVM RBF (AUC={auc:.4f})')
axes[2].plot([0,1],[0,1],'k--',lw=1)
axes[2].fill_between(fpr, tpr, alpha=0.15, color='darkorange')
axes[2].set_xlabel("FPR")
axes[2].set_ylabel("TPR")
axes[2].set_title("ROC Curve")
axes[2].legend()

plt.tight_layout()
plt.savefig("Practical_10_SVM.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_10_SVM.png")
