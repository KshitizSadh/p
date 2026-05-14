# ============================================================
# PRACTICAL 2: Simple Linear Regression
# Dataset: California Housing (sklearn) — uses MedInc → MedHouseVal
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -------------------------------------------------------
# 1. Load Dataset (single feature for Simple LR)
# -------------------------------------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Use MedInc (Median Income) as the single predictor
X = df[['MedInc']]
y = df['MedHouseVal']

print("=" * 60)
print("PRACTICAL 2: Simple Linear Regression")
print("=" * 60)
print(f"\nDataset       : California Housing")
print(f"Feature (X)   : MedInc (Median Income)")
print(f"Target  (y)   : MedHouseVal (Median House Value)")
print(f"Samples       : {len(df)}")

# -------------------------------------------------------
# 2. Train-Test Split (80-20)
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing  Samples : {X_test.shape[0]}")

# -------------------------------------------------------
# 3. Train Model
# -------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nModel Equation : y = {model.coef_[0]:.4f} * MedInc + {model.intercept_:.4f}")

# -------------------------------------------------------
# 4. Evaluation Metrics
# -------------------------------------------------------
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n--- Regression Evaluation (Test Set) ---")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# -------------------------------------------------------
# 5. K-Fold Cross-Validation (k=10)
# -------------------------------------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_r2  = cross_val_score(model, X, y, cv=kf, scoring='r2')
cv_mse = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

print("\n--- 10-Fold Cross-Validation ---")
print(f"CV R² scores  : {np.round(cv_r2, 4)}")
print(f"Mean R²       : {cv_r2.mean():.4f}  |  Std: {cv_r2.std():.4f}")
print(f"Mean CV MSE   : {cv_mse.mean():.4f}  |  Std: {cv_mse.std():.4f}")

# -------------------------------------------------------
# 6. Plots
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Practical 2 – Simple Linear Regression", fontsize=14, fontweight='bold')

# (a) Scatter + Regression Line
sample_idx = np.random.choice(len(X_test), 500, replace=False)
x_vals = X_test.iloc[sample_idx]['MedInc'].values
y_vals = y_test.iloc[sample_idx].values
y_line = model.predict(X_test.iloc[sample_idx])

axes[0].scatter(x_vals, y_vals, alpha=0.3, color='steelblue', s=10, label='Actual')
sorted_idx = np.argsort(x_vals)
axes[0].plot(x_vals[sorted_idx], y_line[sorted_idx], color='red', lw=2, label='Regression Line')
axes[0].set_xlabel("Median Income")
axes[0].set_ylabel("House Value")
axes[0].set_title("Scatter + Regression Line")
axes[0].legend()

# (b) Actual vs Predicted
axes[1].scatter(y_test.iloc[sample_idx], y_line, alpha=0.3, color='darkorange', s=10)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1.5)
axes[1].set_xlabel("Actual Values")
axes[1].set_ylabel("Predicted Values")
axes[1].set_title(f"Actual vs Predicted\n(R²={r2:.4f})")

# (c) Residual Plot
residuals = y_test - y_pred
axes[2].scatter(y_pred, residuals, alpha=0.2, color='purple', s=8)
axes[2].axhline(0, color='red', linestyle='--', lw=1.5)
axes[2].set_xlabel("Predicted Values")
axes[2].set_ylabel("Residuals")
axes[2].set_title("Residual Plot")

plt.tight_layout()
plt.savefig("Practical_02_Simple_Linear_Regression.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_02_Simple_Linear_Regression.png")
