import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv('data/cleaned_dataset.csv')

# Extract features and labels
X_text = df['all_symptoms']
y = df['Disease']  # Ensure column name matches your CSV exactly

# Vectorize symptoms text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X_text)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Create directory for models if not exists
os.makedirs('models', exist_ok=True)

# Save the model and vectorizer using joblib
joblib.dump(model, 'models/disease_predictor.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("✅ Model and vectorizer saved successfully.")
