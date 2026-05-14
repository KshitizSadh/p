# ============================================================
# PRACTICAL 5: Lasso and Ridge Regression
# Dataset: California Housing (all features)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -------------------------------------------------------
# 1. Load & Prepare Dataset
# -------------------------------------------------------
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target
feature_names = list(X.columns)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("PRACTICAL 5: Lasso and Ridge Regression")
print("=" * 60)
print(f"Dataset    : California Housing | Samples: {len(X)}")
print(f"Train/Test : {len(X_train)} / {len(X_test)}")

# -------------------------------------------------------
# 2. Train Models (OLS, Ridge, Lasso)
# -------------------------------------------------------
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
kf = KFold(n_splits=10, shuffle=True, random_state=42)

models = {
    'OLS'  : LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01, max_iter=10000)
}

print("\n--- Model Comparison ---")
print(f"{'Model':>8} | {'MSE':>8} | {'RMSE':>8} | {'MAE':>8} | {'R²':>8} | {'CV R² Mean':>11}")
print("-" * 67)

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred  = m.predict(X_test)
    mse     = mean_squared_error(y_test, y_pred)
    rmse    = np.sqrt(mse)
    mae     = mean_absolute_error(y_test, y_pred)
    r2      = r2_score(y_test, y_pred)
    cv_r2   = cross_val_score(m, X_scaled, y, cv=kf, scoring='r2').mean()
    coefs   = m.coef_ if hasattr(m, 'coef_') else np.zeros(X.shape[1])
    results[name] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'cv_r2': cv_r2,
                     'coef': coefs, 'y_pred': y_pred}
    print(f"{name:>8} | {mse:>8.4f} | {rmse:>8.4f} | {mae:>8.4f} | {r2:>8.4f} | {cv_r2:>11.4f}")

# -------------------------------------------------------
# 3. Alpha Tuning for Ridge & Lasso
# -------------------------------------------------------
print("\n--- Ridge Alpha Tuning ---")
print(f"{'Alpha':>8} | {'CV R²':>8}")
ridge_cv_r2 = []
for alpha in alphas:
    r = Ridge(alpha=alpha)
    s = cross_val_score(r, X_scaled, y, cv=kf, scoring='r2').mean()
    ridge_cv_r2.append(s)
    print(f"{alpha:>8} | {s:>8.4f}")

print("\n--- Lasso Alpha Tuning ---")
print(f"{'Alpha':>8} | {'CV R²':>8} | {'Non-zero coefs':>15}")
lasso_cv_r2 = []
lasso_nz    = []
for alpha in alphas:
    la = Lasso(alpha=alpha, max_iter=10000)
    la.fit(X_train, y_train)
    s  = cross_val_score(la, X_scaled, y, cv=kf, scoring='r2').mean()
    nz = np.sum(la.coef_ != 0)
    lasso_cv_r2.append(s)
    lasso_nz.append(nz)
    print(f"{alpha:>8} | {s:>8.4f} | {nz:>15}")

# -------------------------------------------------------
# 4. Plots
# -------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Practical 5 – Lasso & Ridge Regression", fontsize=14, fontweight='bold')

# (a) Coefficient comparison
x_pos = np.arange(len(feature_names))
width = 0.25
axes[0,0].bar(x_pos - width, results['OLS']['coef'],   width, label='OLS',   color='steelblue')
axes[0,0].bar(x_pos,         results['Ridge']['coef'],  width, label='Ridge', color='darkorange')
axes[0,0].bar(x_pos + width, results['Lasso']['coef'],  width, label='Lasso', color='green')
axes[0,0].set_xticks(x_pos)
axes[0,0].set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
axes[0,0].set_title("Coefficient Comparison")
axes[0,0].legend()
axes[0,0].axhline(0, color='black', lw=0.8)

# (b) Ridge: alpha vs CV R²
axes[0,1].semilogx(alphas, ridge_cv_r2, 'o-', color='darkorange', lw=2)
axes[0,1].set_xlabel("Alpha (log scale)")
axes[0,1].set_ylabel("CV R²")
axes[0,1].set_title("Ridge: Alpha vs CV R²")
axes[0,1].grid(True, which='both', linestyle='--', alpha=0.5)

# (c) Lasso: alpha vs CV R²
axes[0,2].semilogx(alphas, lasso_cv_r2, 'o-', color='green', lw=2)
axes[0,2].set_xlabel("Alpha (log scale)")
axes[0,2].set_ylabel("CV R²")
axes[0,2].set_title("Lasso: Alpha vs CV R²")
axes[0,2].grid(True, which='both', linestyle='--', alpha=0.5)

# (d) Actual vs Predicted (Ridge)
s = np.random.choice(len(y_test), 500, replace=False)
axes[1,0].scatter(y_test.iloc[s], results['Ridge']['y_pred'][s], alpha=0.3, color='darkorange', s=8)
axes[1,0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1.5)
axes[1,0].set_title(f"Ridge: Actual vs Predicted\nR²={results['Ridge']['r2']:.4f}")
axes[1,0].set_xlabel("Actual")
axes[1,0].set_ylabel("Predicted")

# (e) Actual vs Predicted (Lasso)
axes[1,1].scatter(y_test.iloc[s], results['Lasso']['y_pred'][s], alpha=0.3, color='green', s=8)
axes[1,1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1.5)
axes[1,1].set_title(f"Lasso: Actual vs Predicted\nR²={results['Lasso']['r2']:.4f}")
axes[1,1].set_xlabel("Actual")
axes[1,1].set_ylabel("Predicted")

# (f) Lasso non-zero coefficients vs alpha
axes[1,2].semilogx(alphas, lasso_nz, 's-', color='purple', lw=2)
axes[1,2].set_xlabel("Alpha (log scale)")
axes[1,2].set_ylabel("# Non-zero Coefficients")
axes[1,2].set_title("Lasso: Feature Selection vs Alpha")
axes[1,2].grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("Practical_05_Lasso_Ridge.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_05_Lasso_Ridge.png")
