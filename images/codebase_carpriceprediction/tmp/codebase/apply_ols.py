import pickle
import pandas as pd
import statsmodels.api as sm

# Paths inside Docker shared volume
MODEL_PATH = "/tmp/knowledgebase/current_ols_solution.pkl"
DATA_PATH = "/tmp/activationBase/activation_data.csv"

# Load trained OLS model
with open(MODEL_PATH, "rb") as f:
    ols_model = pickle.load(f)

# Load activation data
df = pd.read_csv(DATA_PATH)

# IMPORTANT: add constant (intercept) exactly like training
df_const = sm.add_constant(df, has_constant="add")

# Predict prices
prediction = ols_model.predict(df_const)

print("OLS Price Prediction:", prediction.values)

