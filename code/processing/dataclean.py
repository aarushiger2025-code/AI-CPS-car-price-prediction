import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Load the scraped CSV
df = pd.read_csv("raw_car_data.csv")

brand_dummies = pd.get_dummies(df['brand'], prefix='brand', dtype=int)
df = pd.concat([df.drop(columns=['brand']), brand_dummies], axis=1)

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])

CURRENT_YEAR = 2025
df['age'] = CURRENT_YEAR - df['year']

df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
df = df.dropna(subset=['mileage'])

mileage_scaler = MinMaxScaler()
df['mileage'] = mileage_scaler.fit_transform(df[['mileage']])

df['power_ps'] = pd.to_numeric(df['power_ps'], errors='coerce')
df = df.dropna(subset=['power_ps'])

power_scaler = MinMaxScaler()
df['power_ps'] = power_scaler.fit_transform(df[['power_ps']])

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

fuel_dummies = pd.get_dummies(df['fuel'], prefix='fuel', dtype=int)
df = pd.concat([df.drop(columns=['fuel']), fuel_dummies], axis=1)

df.to_csv("joint_data_collection.csv", index=False)

