import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('HAM10000_metadata.csv')

# Drop rows with missing age values
df = df.dropna(subset=['age'])

# Encode categorical variables
label_encoders = {}
categorical_cols = ['dx','dx_type', 'sex', 'localization']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define a function to inverse transform categorical variables
def inverse_transform_categorical(df, col, encoder):
    df[col] = encoder.inverse_transform(df[col])

# Save preprocessed dataset
df.to_csv('preprocessed_HAM10000.csv', index=False)
