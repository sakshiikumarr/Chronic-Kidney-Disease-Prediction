import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("chronickidneydisease.csv")

# Drop the 'id' column if it exists
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Convert specific numeric columns stored as strings
for col in ['pcv', 'wc', 'rc']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Convert categorical columns to string and lower case
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].astype(str).apply(lambda x: x.str.lower().str.strip())

# Replace target labels with 1 (ckd) and 0 (notckd)
df['classification'] = df['classification'].replace({'ckd': 1, 'notckd': 0})

# Fill missing categorical values with mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Label encode all categorical features
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('classification', axis=1)
y = df['classification']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save the model
with open("gradient_boosting_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'gradient_boosting_model.pkl'")
