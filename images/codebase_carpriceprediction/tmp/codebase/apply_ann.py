import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Paths inside Docker volume
MODEL_PATH = "/tmp/knowledgeBase/current_ann_solution.keras"
DATA_PATH = "/tmp/activationBase/activation_data.csv"

# Load trained ANN model
model = load_model(MODEL_PATH)

# Load activation data
df = pd.read_csv(DATA_PATH)

# Predict (log-price space if you trained with log1p)
y_pred_log = model.predict(df).flatten()

# Convert back to Euro space
y_pred = np.expm1(y_pred_log)

print("ANN Price Prediction:", y_pred)
