# test_exp.py
import requests

# Example: 8 features for California housing dataset
features = [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]
payload = {"features": features}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())