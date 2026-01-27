from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the processed dataset
df = pd.read_csv("joint_data_collection.csv")

# Select columns to be used for splitting
selected_columns = (
    ['mileage', 'year', 'age', 'power_ps', 'price'] +
    [col for col in df.columns if col.startswith('brand_')]
)

df_model = df[selected_columns]

# Train / test split
train_df, test_df = train_test_split(
    df_model,
    test_size=0.2,
    random_state=42
)

# Save datasets
train_df.to_csv("training_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

# Activation sample (single row from test set)
activation_df = test_df.sample(n=1, random_state=42)
activation_df.drop(columns=['price']).to_csv(
    "activation_data.csv", index=False
)

