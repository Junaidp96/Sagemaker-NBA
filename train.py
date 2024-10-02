import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Get the input path for the dataset
input_data_path = os.environ.get('SM_CHANNEL_TRAINING', 's3://nbaallstar/nba.csv')  # Local fallback
model_output_path = os.environ.get('SM_MODEL_DIR', 's3://nbaallstar/output/')  # Local fallback

print(f"Input data path: {input_data_path}")
print(f"Model output path: {model_output_path}")

# Load dataset
try:
    df = pd.read_csv(os.path.join(input_data_path, 'nba_file.csv'))
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Preprocess and split data
X = df.drop(columns=['Salary'])  # Adjust 'target' to your actual target column
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model training completed.")

# Save the model
joblib.dump(model, os.path.join(model_output_path, "model.joblib"))
print(f"Model saved to {model_output_path}")

