# ============================================================
# PRACTICAL 3: Multiple Linear Regression
# Dataset: California Housing (all 8 features)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

print("=" * 60)
print("PRACTICAL 3: Multiple Linear Regression")
print("=" * 60)
print(f"\nDataset    : California Housing")
print(f"Features   : {list(X.columns)}")
print(f"Samples    : {len(X)}")
print(f"\nFeature Statistics:")
print(X.describe().round(2).to_string())

# -------------------------------------------------------
# 2. Feature Scaling + Train-Test Split
# -------------------------------------------------------
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing  Samples : {X_test.shape[0]}")

# -------------------------------------------------------
# 3. Train Model
# -------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- Model Coefficients ---")
for feat, coef in zip(X.columns, model.coef_):
    print(f"  {feat:20s}: {coef:+.4f}")
print(f"  {'Intercept':20s}: {model.intercept_:+.4f}")

# -------------------------------------------------------
# 4. Evaluation Metrics
# -------------------------------------------------------
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

# Adjusted R²
n = X_test.shape[0]
p = X_test.shape[1]
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\n--- Regression Evaluation (Test Set) ---")
print(f"MSE          : {mse:.4f}")
print(f"RMSE         : {rmse:.4f}")
print(f"MAE          : {mae:.4f}")
print(f"R²           : {r2:.4f}")
print(f"Adjusted R²  : {r2_adj:.4f}")

# -------------------------------------------------------
# 5. K-Fold Cross-Validation (k=10)
# -------------------------------------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_r2  = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
cv_mse = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')

print("\n--- 10-Fold Cross-Validation ---")
print(f"CV R² scores  : {np.round(cv_r2, 4)}")
print(f"Mean R²       : {cv_r2.mean():.4f}  |  Std: {cv_r2.std():.4f}")
print(f"Mean CV MSE   : {cv_mse.mean():.4f}  |  Std: {cv_mse.std():.4f}")

# -------------------------------------------------------
# 6. Plots
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Practical 3 – Multiple Linear Regression", fontsize=14, fontweight='bold')

# (a) Actual vs Predicted
sample_idx = np.random.choice(len(y_test), 1000, replace=False)
axes[0].scatter(y_test[sample_idx], y_pred[sample_idx], alpha=0.3, color='steelblue', s=8)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1.5)
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Actual vs Predicted\n(R²={r2:.4f})")

# (b) Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.2, color='darkorange', s=6)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot")

# (c) Feature Coefficients
coef_df = pd.Series(model.coef_, index=X.columns).sort_values()
coef_df.plot(kind='barh', ax=axes[2], color=['red' if c < 0 else 'green' for c in coef_df])
axes[2].axvline(0, color='black', lw=0.8)
axes[2].set_title("Feature Coefficients (Scaled)")
axes[2].set_xlabel("Coefficient Value")

plt.tight_layout()
plt.savefig("Practical_03_Multiple_Linear_Regression.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_03_Multiple_Linear_Regression.png")
