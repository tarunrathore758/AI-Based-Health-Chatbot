import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "symptoms": "fever,cough,headache"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Predicted Disease:", result['disease'])
    print("Recommended Doctor:", result['doctor'])
else:
    print("Error:", response.status_code, response.text)
