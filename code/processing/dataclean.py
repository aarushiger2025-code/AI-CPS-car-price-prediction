"""
Cleaning raw dataset 

This Script:
   - Loads raw scraped car listing data from a CSV
   - Cleans and converts numeric columns (year, mileage, power, price)
   - Engineers new features (car age)
   - Removes extreme price outliers using an upper cap
   - Removes remaining outliers using IQR method
   - One-hot encodes categorical variables (brand, fuel)
   - Scales mileage and power to a 0–1 range
   - Outputs a clean, model-ready dataset as a new CSV
"""

# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load Raw Dataset
df = pd.read_csv("raw_car_data.csv")

# Drop duplicates if any
df = df.drop_duplicates()


# Clean numeric columns

df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
df = df.dropna(subset=['mileage'])

df['power_ps'] = pd.to_numeric(df['power_ps'], errors='coerce')
df = df.dropna(subset=['power_ps'])

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

# Clean Year & Create Age Feature

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])

CURRENT_YEAR = 2026
df['age'] = CURRENT_YEAR - df['year']

# Only keep realistic car ages
df = df[(df['age'] >= 0) & (df['age'] <= 40)]


# HARD UPPER PRICE CAP

PRICE_CAP = 150000
df = df[df['price'] <= PRICE_CAP]


# Remove Outliers (IQR Method)

def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    return df[
        (df[col] >= Q1 - 1.5 * IQR) &
        (df[col] <= Q3 + 1.5 * IQR)
    ]

df = remove_outliers_iqr(df, 'price')
df = remove_outliers_iqr(df, 'mileage')


# Categorical Encoding
# One-Hot Encode Car Brand
brand_dummies = pd.get_dummies(df['brand'], prefix='brand', dtype=int)
df = pd.concat([df.drop(columns=['brand']), brand_dummies], axis=1)

# One-Hot Encode Fuel Type
fuel_dummies = pd.get_dummies(df['fuel'], prefix='fuel', dtype=int)
df = pd.concat([df.drop(columns=['fuel']), fuel_dummies], axis=1)


# Feature Scaling

mileage_scaler = MinMaxScaler()
df['mileage'] = mileage_scaler.fit_transform(df[['mileage']])

power_scaler = MinMaxScaler()
df['power_ps'] = power_scaler.fit_transform(df[['power_ps']])


# Save Clean Dataset

df.to_csv("joint_data_collection.csv", index=False)

print("Data cleaning complete. Clean dataset saved as joint_data_collection.csv")
