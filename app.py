import pickle
import json
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load disease-to-doctor map
with open('disease_doctor_map.json') as f:
    disease_doctor_map = json.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    doctor = None

    if request.method == 'POST':
        symptoms = request.form['symptoms']
        symptoms_vector = vectorizer.transform([symptoms])
        prediction = model.predict(symptoms_vector)[0]
        doctor = disease_doctor_map.get(prediction, "General Physician")

    return render_template('index.html', prediction=prediction, doctor=doctor)

if __name__ == '__main__':
    app.run(debug=True)
