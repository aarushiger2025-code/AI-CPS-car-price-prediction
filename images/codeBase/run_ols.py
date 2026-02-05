"""

Training an OLS regression model for second-hand car price prediction

This script:
  - Loads preprocessed training & testing datasets
  - Trains an Ordinary Least Squares (OLS) regression model
  - Evaluates predictive performance on test data
  - Saves the trained model as a .pkl file
  - Exports a full statistical summary report
  - Generates classical OLS diagnostic plots
  - Helps validate regression assumptions

"""

# IMPORT REQUIRED LIBRARIES

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# CONFIGURATION SECTION
# Dataset paths

TRAIN_PATH = r"C:\Users\Abhay\OneDrive\Desktop\Sign_Lang_Codes\Car_price_prediction\training_data.csv"
TEST_PATH = r"C:\Users\Abhay\OneDrive\Desktop\Sign_Lang_Codes\Car_price_prediction\test_data.csv"

# Output directories

PLOT_DIR = r"C:\Users\Abhay\OneDrive\Desktop\Sign_Lang_Codes\Car_price_prediction\plots_ols"
MODEL_DIR = r"C:\Users\Abhay\OneDrive\Desktop\Sign_Lang_Codes\Car_price_prediction\models"

# Create folders if missing

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# LOAD DATASETS

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)


# SPLIT FEATURES & TARGET
# Separate predictors and target variable

X_train = train_df.drop(columns=["price"])
y_train = train_df["price"]

X_test = test_df.drop(columns=["price"])
y_test = test_df["price"]

# Add intercept (constant term) required for OLS

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# TRAIN OLS MODEL

print("Training OLS regression model...")

ols_model = sm.OLS(y_train, X_train_const).fit()

print("Training complete.")


# SAVE MODEL & SUMMARY
# Save model as pickle file

PKL_PATH = os.path.join(MODEL_DIR, "current_ols_solution.pkl")

with open(PKL_PATH, "wb") as f:
    pickle.dump(ols_model, f)

# Save full regression summary

SUMMARY_PATH = os.path.join(MODEL_DIR, "ols_summary.txt")

with open(SUMMARY_PATH, "w") as f:
    f.write(ols_model.summary().as_text())

print(f"\nOLS model saved:")
print(f"- Model   : {PKL_PATH}")
print(f"- Summary : {SUMMARY_PATH}")


# TEST SET EVALUATION

print("\nEvaluating on test data...")

# Generate predictions

y_pred = ols_model.predict(X_test_const)

# Regression metrics

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- OLS Test Performance ---")
print(f"MAE  : {mae:.2f} €")
print(f"RMSE : {rmse:.2f} €")
print(f"R²   : {r2:.3f}")


# OLS DIAGNOSTICS (TRAINING DATA)
# Fitted values & residuals

fitted = ols_model.fittedvalues
residuals = ols_model.resid

# Influence statistics

influence = ols_model.get_influence()

leverage = influence.hat_matrix_diag
std_residuals = influence.resid_studentized_internal
cooks_d = influence.cooks_distance[0]

# Remove invalid / infinite rows

mask = (
    np.isfinite(fitted) &
    np.isfinite(residuals) &
    np.isfinite(leverage) &
    np.isfinite(std_residuals) &
    np.isfinite(cooks_d)
)

fitted = fitted[mask]
residuals = residuals[mask]
leverage = leverage[mask]
std_residuals = std_residuals[mask]
cooks_d = cooks_d[mask]


# RESIDUALS VS FITTED

plt.figure(figsize=(8, 6))

plt.scatter(fitted, residuals, alpha=0.5, color="steelblue")
plt.axhline(0, color="red", linestyle="--", linewidth=2)

plt.xlabel("Fitted Values (€)")
plt.ylabel("Residuals (€)")
plt.title("Residuals vs Fitted (OLS)")
plt.grid(True)

plt.savefig(os.path.join(PLOT_DIR, "residuals_vs_fitted.png"))
plt.close()

# SCALE–LOCATION PLOT

plt.figure(figsize=(8, 6))

plt.scatter(
    fitted,
    np.sqrt(np.abs(std_residuals)),
    alpha=0.5,
    color="darkorange"
)

plt.xlabel("Fitted Values (€)")
plt.ylabel("√|Standardized Residuals|")
plt.title("Scale-Location Plot (OLS)")
plt.grid(True)

plt.savefig(os.path.join(PLOT_DIR, "scale_location.png"))
plt.close()


# NORMAL Q-Q PLOT

plt.figure(figsize=(8, 6))

(osm, osr), (slope, intercept, _) = stats.probplot(
    residuals,
    dist="norm"
)

plt.scatter(osm, osr, color="steelblue", alpha=0.6)
plt.plot(osm, slope * osm + intercept, color="red", linewidth=2)

plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Residuals")
plt.title("Normal Q-Q Plot (OLS)")
plt.grid(True)

plt.savefig(os.path.join(PLOT_DIR, "qq_plot.png"))
plt.close()


# RESIDUALS VS LEVERAGE (COOK’S DISTANCE)

plt.figure(figsize=(9, 7))

# Bubble plot (size proportional to Cook’s Distance)

plt.scatter(
    leverage,
    std_residuals,
    s=1000 * cooks_d,
    alpha=0.6,
    color="steelblue",
    edgecolor="black"
)

plt.axhline(0, color="red", linestyle="--", linewidth=2)

# Cook’s distance contours

p = X_train_const.shape[1]
h_vals = np.linspace(0.1, leverage.max(), 100)

for D in [0.5, 1]:
    bound = np.sqrt((D * p * (1 - h_vals)) / h_vals)

    plt.plot(h_vals, bound, "r--", linewidth=1)
    plt.plot(h_vals, -bound, "r--", linewidth=1)

    plt.text(
        h_vals[-1],
        bound[-1],
        f"Cook's D = {D}",
        color="red",
        fontsize=9
    )

plt.xlabel("Leverage")
plt.ylabel("Standardized Residuals")
plt.title("Residuals vs Leverage
