"""
Cleaning raw dataset 


This Script:

   - Loads raw scraped car listing data from a CSV
   - Cleans and converts numeric columns (year, mileage, power, price)
   - Engineers new features (car age)
   - One-hot encodes categorical variables (brand, fuel)
   - Scales mileage and power to a 0–1 range
   - Outputs a clean, model-ready dataset as a new CSV

"""

#import Required Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Load the scraped raw car CSV

df = pd.read_csv("raw_car_data.csv")

# One-hot encode car brands
# This turns a single 'brand' column into multiple binary columns

brand_dummies = pd.get_dummies(df['brand'], prefix='brand', dtype=int)
df = pd.concat([df.drop(columns=['brand']), brand_dummies], axis=1)

# Clean and process the 'year' column
# numeric conversion if any string
# Drop rows with invalid/missing years
# Create a new 'age' feature

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])

CURRENT_YEAR = 2025
df['age'] = CURRENT_YEAR - df['year']

# Clean and normalize mileage and engine Power (PS)
# - Convert to numeric 
# - Drop rows with invalid/missing mileage and engine Power (PS) values
# - Scale values between 0 and 1

df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
df = df.dropna(subset=['mileage'])

mileage_scaler = MinMaxScaler()
df['mileage'] = mileage_scaler.fit_transform(df[['mileage']])

df['power_ps'] = pd.to_numeric(df['power_ps'], errors='coerce')
df = df.dropna(subset=['power_ps'])

power_scaler = MinMaxScaler()
df['power_ps'] = power_scaler.fit_transform(df[['power_ps']])

# Clean price column
# - Convert to numeric if any string
# - Drop rows with missing or invalid prices

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

# One-hot encode fuel type

fuel_dummies = pd.get_dummies(df['fuel'], prefix='fuel', dtype=int)
df = pd.concat([df.drop(columns=['fuel']), fuel_dummies], axis=1)

# Save the final cleaned dataset

df.to_csv("joint_data_collection.csv", index=False)

