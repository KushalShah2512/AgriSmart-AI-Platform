import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Crop_recommendation.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'crop_recommendation_model.pkl')

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Random Forest model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Training Completed! Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, MODEL_PATH)
print(f"Model Successfully saved to {MODEL_PATH}")
