import requests

# GET request
get_response = requests.get("http://127.0.0.1:8000/")
print("Status Code:", get_response.status_code)
print("Result:", get_response.json()["message"])

# POST request
data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

post_response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=data
)

print("Status Code:", post_response.status_code)
print("Result:", post_response.json()["result"])
