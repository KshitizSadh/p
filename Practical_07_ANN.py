# ============================================================
# PRACTICAL 7: Artificial Neural Network (MLP)
# Dataset: Breast Cancer Wisconsin (classification)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
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
print("PRACTICAL 7: Artificial Neural Network (MLP)")
print("=" * 60)
print(f"\nDataset : Breast Cancer Wisconsin")
print(f"Classes : {list(data.target_names)}")
print(f"Samples : {len(X)}")

# -------------------------------------------------------
# 2. Preprocess + Split
# -------------------------------------------------------
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# -------------------------------------------------------
# 3. Define & Train MLP
# Architecture: Input(30) -> 64 -> 32 -> Output(2)
# -------------------------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    verbose=False
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\nTraining stopped after {model.n_iter_} iterations (early stopping)")

# -------------------------------------------------------
# 4. Evaluation Metrics
# -------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

accuracy    = accuracy_score(y_test, y_pred)
error_rate  = 1 - accuracy
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)
specificity = TN / (TN + FP)
f1          = 2 * precision * recall / (precision + recall)
auc         = roc_auc_score(y_test, y_prob)

print("\n--- Confusion Matrix ---")
print(f"  TN={TN}  FP={FP}")
print(f"  FN={FN}  TP={TP}")
print("\n--- Classification Metrics ---")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Error Rate  : {error_rate:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"AUC-ROC     : {auc:.4f}")

print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# -------------------------------------------------------
# 5. K-Fold Cross-Validation (k=5, ANN is slow)
# -------------------------------------------------------
cv_model = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu',
                          solver='adam', max_iter=300, random_state=42)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(cv_model, X_scaled, y, cv=kf, scoring='accuracy')
cv_auc = cross_val_score(cv_model, X_scaled, y, cv=kf, scoring='roc_auc')

print("--- 5-Fold Cross-Validation ---")
print(f"CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"CV AUC-ROC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# -------------------------------------------------------
# 6. Plots
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Practical 7 – Artificial Neural Network (MLP)", fontsize=14, fontweight='bold')

# (a) Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names).plot(
    ax=axes[0], colorbar=False, cmap='Purples')
axes[0].set_title("Confusion Matrix")

# (b) Training Loss Curve
axes[1].plot(model.loss_curve_, color='blue', lw=2, label='Train Loss')
if hasattr(model, 'validation_scores_') and model.validation_scores_ is not None:
    axes[1].plot(model.validation_scores_, color='orange', lw=2, linestyle='--', label='Val Score')
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss / Score")
axes[1].set_title("Loss Curve")
axes[1].legend()

# (c) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={auc:.4f}')
axes[2].plot([0,1],[0,1],'k--',lw=1)
axes[2].fill_between(fpr, tpr, alpha=0.15, color='darkorange')
axes[2].set_xlabel("FPR")
axes[2].set_ylabel("TPR")
axes[2].set_title("ROC Curve")
axes[2].legend()

plt.tight_layout()
plt.savefig("Practical_07_ANN.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_07_ANN.png")
