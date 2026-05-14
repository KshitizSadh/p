# ============================================================
# PRACTICAL 4: Polynomial Regression
# Dataset: Auto MPG (UCI) — horsepower → mpg
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -------------------------------------------------------
# 1. Load Auto MPG Dataset (UCI via URL)
# -------------------------------------------------------
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "auto-mpg/auto-mpg.data")

col_names = ['mpg','cylinders','displacement','horsepower',
             'weight','acceleration','model_year','origin','car_name']

df = pd.read_csv(url, names=col_names, sep='\s+', na_values='?')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Single predictor: horsepower
X = df[['horsepower']].values
y = df['mpg'].values

print("=" * 60)
print("PRACTICAL 4: Polynomial Regression")
print("=" * 60)
print(f"\nDataset  : Auto MPG (UCI)")
print(f"Feature  : horsepower")
print(f"Target   : mpg")
print(f"Samples  : {len(df)}")

# -------------------------------------------------------
# 2. Train-Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining : {len(X_train)} | Testing : {len(X_test)}")

# -------------------------------------------------------
# 3. Compare Polynomial Degrees 1–5
# -------------------------------------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = {}

print("\n--- Degree Comparison ---")
print(f"{'Degree':>7} | {'Train R²':>9} | {'Test R²':>8} | {'Test MSE':>9} | {'CV R² Mean':>11} | {'CV R² Std':>10}")
print("-" * 67)

for deg in range(1, 6):
    pipe = Pipeline([
        ('poly',   PolynomialFeatures(degree=deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr',     LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2  = r2_score(y_test,  y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    cv_r2    = cross_val_score(pipe, X, y, cv=kf, scoring='r2')

    results[deg] = {
        'pipeline': pipe, 'train_r2': train_r2, 'test_r2': test_r2,
        'test_mse': test_mse, 'cv_mean': cv_r2.mean(), 'cv_std': cv_r2.std(),
        'y_pred_test': y_pred_test
    }
    print(f"{deg:>7} | {train_r2:>9.4f} | {test_r2:>8.4f} | {test_mse:>9.4f} | "
          f"{cv_r2.mean():>11.4f} | {cv_r2.std():>10.4f}")

# Best model (degree 2 typically best for horsepower)
best_deg = max(results, key=lambda d: results[d]['cv_mean'])
best = results[best_deg]
print(f"\nBest Degree (by CV R²): {best_deg}")
print(f"Test R²  : {best['test_r2']:.4f}")
print(f"Test MSE : {best['test_mse']:.4f}")
print(f"Test MAE : {mean_absolute_error(y_test, best['y_pred_test']):.4f}")

# -------------------------------------------------------
# 4. Plots
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Practical 4 – Polynomial Regression", fontsize=14, fontweight='bold')

# (a) Fit curves for degrees 1,2,3
colors = ['blue', 'green', 'red']
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
axes[0].scatter(X, y, alpha=0.3, color='gray', s=10, label='Data')
for deg, color in zip([1, 2, 3], colors):
    y_range = results[deg]['pipeline'].predict(X_range)
    axes[0].plot(X_range, y_range, color=color, lw=2, label=f'Deg {deg}')
axes[0].set_xlabel("Horsepower")
axes[0].set_ylabel("MPG")
axes[0].set_title("Polynomial Fits (Deg 1,2,3)")
axes[0].legend()

# (b) Actual vs Predicted (best model)
axes[1].scatter(y_test, best['y_pred_test'], alpha=0.5, color='darkorange', s=20)
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1.5)
axes[1].set_xlabel("Actual MPG")
axes[1].set_ylabel("Predicted MPG")
axes[1].set_title(f"Actual vs Predicted (Deg {best_deg})\nR²={best['test_r2']:.4f}")

# (c) R² vs Degree
degrees  = list(results.keys())
test_r2s = [results[d]['test_r2'] for d in degrees]
cv_r2s   = [results[d]['cv_mean'] for d in degrees]
axes[2].plot(degrees, test_r2s, 'o-', color='steelblue', label='Test R²')
axes[2].plot(degrees, cv_r2s,   's--', color='red',       label='CV R² (Mean)')
axes[2].set_xlabel("Polynomial Degree")
axes[2].set_ylabel("R²")
axes[2].set_title("R² vs Polynomial Degree")
axes[2].legend()
axes[2].set_xticks(degrees)

plt.tight_layout()
plt.savefig("Practical_04_Polynomial_Regression.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as Practical_04_Polynomial_Regression.png")
