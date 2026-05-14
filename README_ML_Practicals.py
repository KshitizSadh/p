# ============================================================
# ML Lab Practicals – Requirements & Quick Reference
# ============================================================

# ---------- requirements.txt ----------
# Run:  pip install -r requirements.txt

scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0

# ============================================================
# README – All 12 Practicals Summary
# ============================================================

"""
MACHINE LEARNING LAB – PRACTICAL REFERENCE
===========================================

All practicals use Python + scikit-learn. Each file is self-contained
and runnable independently.

─────────────────────────────────────────────────────────────────────
 #   File                                    Algorithm         Dataset
─────────────────────────────────────────────────────────────────────
 1   Practical_01_Naive_Bayes.py             Naïve Bayes       Iris
 2   Practical_02_Simple_Linear_Regression   Simple LR         California Housing (MedInc → Price)
 3   Practical_03_Multiple_Linear_Regression Multiple LR       California Housing (All 8 features)
 4   Practical_04_Polynomial_Regression      Polynomial Reg.   Auto MPG (UCI)
 5   Practical_05_Lasso_Ridge_Regression     Lasso & Ridge     California Housing
 6   Practical_06_Logistic_Regression        Logistic Reg.     Breast Cancer Wisconsin
 7   Practical_07_ANN.py                     MLP / ANN         Breast Cancer Wisconsin
 8   Practical_08_KNN.py                     K-NN Classifier   Iris
 9   Practical_09_Decision_Tree.py           Decision Tree     Breast Cancer Wisconsin
10   Practical_10_SVM.py                     SVM               Breast Cancer Wisconsin
11   Practical_11_KMeans_Clustering.py       K-Means           Customer Segmentation (synthetic)
12   Practical_12_Hierarchical_Clustering.py Hierarchical      Customer Segmentation (synthetic)
─────────────────────────────────────────────────────────────────────

METRICS REPORTED
────────────────
Regression  (P2, P3, P4, P5): MSE, RMSE, MAE, R², Adjusted R², CV scores
Classification (P1,P6–P10)  : Accuracy, TP, TN, FP, FN, Error Rate,
                               Precision, Recall, Specificity, F1, AUC-ROC
Clustering  (P11, P12)       : Inertia/WCSS, Silhouette Score,
                               Davies-Bouldin Score, Calinski-Harabasz Index

EVALUATION STRATEGY (per practical)
─────────────────────────────────────
• Train-Test Split : 80% train / 20% test (stratified for classification)
• Cross-Validation : 10-Fold CV for all (5-Fold for ANN due to compute)
• Hyperparameter   : Grid Search (SVM), Elbow/Silhouette (KNN, K-Means,
  Tuning             Hierarchical), Degree sweep (Polynomial), Alpha sweep
                     (Lasso/Ridge), Max-depth sweep (Decision Tree)

HOW TO RUN
──────────
1. Install dependencies:
     pip install scikit-learn numpy pandas matplotlib seaborn scipy

2. Run any practical:
     python Practical_01_Naive_Bayes.py

3. Datasets are auto-downloaded via sklearn or URL.
   Only Auto MPG (P4) fetches from UCI; internet connection required.
   If offline, replace with any regression CSV.

4. For Practicals 11 & 12, place 'Mall_Customers.csv' in the same
   directory for real data, or the code will generate synthetic data.

NOTES FOR REPORT WRITING
─────────────────────────
• All console output (metrics) is printed clearly — copy into report.
• Each practical saves PNG plot(s) automatically.
• Mention dataset source in report:
    - Iris, Breast Cancer, California Housing → sklearn.datasets
    - Auto MPG → https://archive.ics.uci.edu/ml/datasets/auto+mpg
    - Mall Customers → Kaggle / synthetic
"""
