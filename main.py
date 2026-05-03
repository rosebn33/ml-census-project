from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from ml.model import load_model, inference
from ml.data import process_data


app = FastAPI()


# Load model and preprocessing objects
model, encoder, lb = load_model()


# Categorical features (must match training)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
def read_root():
    return {"message": "Hello from the API!"}


@app.post("/predict")
def predict(data: CensusData):
    df = pd.DataFrame(
        [
            {
                "age": data.age,
                "workclass": data.workclass,
                "fnlwgt": data.fnlwgt,
                "education": data.education,
                "education-num": data.education_num,
                "marital-status": data.marital_status,
                "occupation": data.occupation,
                "relationship": data.relationship,
                "race": data.race,
                "sex": data.sex,
                "capital-gain": data.capital_gain,
                "capital-loss": data.capital_loss,
                "hours-per-week": data.hours_per_week,
                "native-country": data.native_country,
            }
        ]
    )

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)[0]
    result = ">50K" if pred == 1 else "<=50K"

    return {"result": result}
