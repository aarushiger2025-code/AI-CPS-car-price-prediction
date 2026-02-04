"""

Training an ANN model for second-hand car price prediction

This script:
  - Loads preprocessed training & testing datasets
  - Splits features and target (price)
  - Applies log transformation to stabilize price variance
  - Builds and trains an Artificial Neural Network (ANN)
  - Uses early stopping to avoid overfitting
  - Evaluates model performance with regression metrics
  - Generates training + diagnostic plots
  - Saves the trained model 

"""


# import required libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# Paths
# Absolute paths to training and testing datasets

TRAIN_PATH = r"C:\Users\Abhay\OneDrive\Desktop\Sign_Lang_Codes\Car-price-prediction\training_data.csv"
TEST_PATH = r"C:\Users\Abhay\OneDrive\Desktop\Sign_Lang_Codes\Car-price-prediction\test_data.csv"

# Directories for saving plots and trained models

PLOT_DIR = r"C:\Users\Abhay\OneDrive\Desktop\Sign_Lang_Codes\Car-price-prediction\plots"
MODEL_DIR = r"C:\Users\Abhay\OneDrive\Desktop\Sign_Lang_Codes\Car-price-prediction\models"

# Create directories if they don’t already exist 

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Final model save path

MODEL_PATH = os.path.join(MODEL_DIR, "car_price_ann.keras")


# Load data
# Read processed training and testing CSV files

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# Split features and target
# Separate predictors (X) and target (y = price)

X_train = train_df.drop(columns=["price"])
X_test = test_df.drop(columns=["price"])

# Log-transform price to stabilize variance & reduce skew

# log1p handles zero values safely
y_train = np.log1p(train_df["price"])
y_test = np.log1p(test_df["price"])

# Number of input features for ANN input layer

input_dim = X_train.shape[1]

# Building ANN model
# Simple feedforward deep neural network

model = Sequential([
    Dense(256, activation="relu", input_shape=(input_dim,)),  # Input + hidden layer
    Dense(128, activation="relu"),                            # Hidden layer
    Dense(64, activation="relu"),                             # Hidden layer
    Dense(1)                                                  # Output layer (regression)
])

# Compile model with optimizer + loss + metrics

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",   # Regression loss
    metrics=["mae"]
)

# Early stopping
# Stops training when validation loss stops improving

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,              # Wait 10 epochs before stopping
    restore_best_weights=True
)

# Train
# Train model with validation split

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,   # 20% of training data used for validation
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Predict
# Predict log-prices

y_pred_log = model.predict(X_test).flatten()

# Convert predictions back to real price scale

y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)


# Metrics
# Standard regression evaluation metrics

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n--- Final ANN Test Performance ---")
print(f"MAE  : {mae:.2f} €")
print(f"RMSE : {rmse:.2f} €")
print(f"R²   : {r2:.3f}")


# Residual calculations
# Residual = Actual − Predicted

residuals = y_true - y_pred

# Standardized residuals for diagnostic plots

std_residuals = (residuals - residuals.mean()) / residuals.std()

# Fitted values = predictions

fitted = y_pred

# Training History Plots

# Loss curve (Train vs Validation)

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE (log-price)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "loss_curve.png"))
plt.close()

# MAE curve

plt.figure()
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE (log-price)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "mae_curve.png"))
plt.close()


# Prediction Performance Plots

# Predicted vs Actual prices

plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.5, color="steelblue")

# Perfect prediction reference line

plt.plot(
    [y_true.min(), y_true.max()],
    [y_true.min(), y_true.max()],
    color="red",
    linewidth=2
)

plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("Predicted vs Actual Prices")
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "predicted_vs_actual.png"))
plt.close()


# Residual Diagnostic Plots

# Residuals vs Fitted

plt.figure(figsize=(8, 6))
plt.scatter(fitted, residuals, alpha=0.5, color="steelblue")
plt.axhline(0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Fitted Values (€)")
plt.ylabel("Residuals (€)")
plt.title("Residuals vs Fitted Values")
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "residuals_vs_fitted.png"))
plt.close()

# Scale-Location plot 

plt.figure(figsize=(8, 6))
plt.scatter(
    fitted,
    np.sqrt(np.abs(std_residuals)),
    alpha=0.5,
    color="steelblue"
)
plt.xlabel("Fitted Values (€)")
plt.ylabel("√|Standardized Residuals|")
plt.title("Scale-Location Plot")
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "scale_location.png"))
plt.close()

# Normal Q-Q plot (checks residual normality)

plt.figure(figsize=(8, 6))

(osm, osr), (slope, intercept, r) = stats.probplot(
    residuals,
    dist="norm"
)

plt.scatter(osm, osr, color="steelblue", alpha=0.6)
plt.plot(osm, slope * osm + intercept, color="red", linewidth=2)

plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Residuals")
plt.title("Normal Q-Q Plot")
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "qq_plot.png"))
plt.close()

# Residuals vs Leverage (approximation for ANN)
# True leverage is defined for linear models

leverage = np.sum(X_test.values ** 2, axis=1)
leverage = leverage / leverage.max()

plt.figure(figsize=(8, 6))
plt.scatter(leverage, std_residuals, alpha=0.5, color="steelblue")
plt.axhline(0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Leverage (normalized)")
plt.ylabel("Standardized Residuals")
plt.title("Residuals vs Leverage")
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "residuals_vs_leverage.png"))
plt.close()

# Error distribution histogram

plt.figure()
plt.hist(residuals, bins=50, color="slateblue")
plt.xlabel("Prediction Error (€)")
plt.ylabel("Count")
plt.title("Error Distribution")
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "error_distribution.png"))
plt.close()



# Save trained model
# Saves model in Keras native format

model.save(MODEL_PATH)
print(f"\nModel saved successfully at:\n{MODEL_PATH}")
