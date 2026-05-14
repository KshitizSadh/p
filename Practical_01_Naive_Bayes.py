# ============================================================
# PRACTICAL 1: Naïve Bayes Classifier
# Dataset: Iris Dataset (UCI ML Repository)
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize

# -------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

print("=" * 60)
print("PRACTICAL 1: Naïve Bayes Classifier")
print("=" * 60)
print(f"\nDataset Shape : {X.shape}")
print(f"Classes       : {list(iris.target_names)}")
print(f"Class Distribution:\n{y.value_counts().to_string()}")

# -------------------------------------------------------
# 2. Train-Test Split (80-20)
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing  Samples : {X_test.shape[0]}")

# -------------------------------------------------------
# 3. Train Naïve Bayes Model
# -------------------------------------------------------
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------------------------------------------
# 4. Evaluation Metrics
# -------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)

print("\n--- Confusion Matrix ---")
print(cm)

# Per-class metrics from report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Aggregate metrics (macro average)
precision = report['macro avg']['precision']
recall    = report['macro avg']['recall']
f1        = report['macro avg']['f1-score']

# TP, TN, FP, FN from confusion matrix (for each class, one-vs-rest)
print("--- Per-Class TP / TN / FP / FN ---")
n_classes = len(iris.target_names)
for i, cls in enumerate(iris.target_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - TP - FP - FN
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    print(f"  {cls:12s} | TP={TP}  TN={TN}  FP={FP}  FN={FN}  "
          f"Specificity={specificity:.4f}")

print(f"\nOverall Accuracy  : {accuracy:.4f}")
print(f"Macro Precision   : {precision:.4f}")
print(f"Macro Recall      : {recall:.4f}")
print(f"Macro F1-Score    : {f1:.4f}")
print(f"Error Rate        : {1 - accuracy:.4f}")

# AUC (One-vs-Rest for multi-class)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_prob     = model.predict_proba(X_test)
auc        = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
print(f"AUC (OvR macro)   : {auc:.4f}")

# -------------------------------------------------------
# 5. K-Fold Cross-Validation (k=10)
# -------------------------------------------------------
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"\n--- 10-Fold Cross-Validation ---")
print(f"Fold Accuracies : {np.round(cv_scores, 4)}")
print(f"Mean Accuracy   : {cv_scores.mean():.4f}")
print(f"Std Dev         : {cv_scores.std():.4f}")

# -------------------------------------------------------
# 6. Plots
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Practical 1 – Naïve Bayes Classifier", fontsize=14, fontweight='bold')

# (a) Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix")

# (b) ROC Curves (per class)
colors = ['darkorange', 'green', 'blue']
for i, (cls, color) in enumerate(zip(iris.target_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    axes[1].plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC={roc_auc_score(y_test_bin[:,i], y_prob[:,i]):.2f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curves (OvR)")
axes[1].legend(fontsize=8)

# (c) Cross-validation accuracy
axes[2].bar(range(1, 11), cv_scores, color='steelblue', edgecolor='navy')
axes[2].axhline(cv_scores.mean(), color='red', linestyle='--', label=f'Mean={cv_scores.mean():.3f}')
axes[2].set_xlabel("Fold")
axes[2].set_ylabel("Accuracy")
axes[2].set_title("10-Fold CV Accuracy")
axes[2].legend()

plt.tight_layout()
plt.savefig("Practical_01_Naive_Bayes.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_01_Naive_Bayes.png")
