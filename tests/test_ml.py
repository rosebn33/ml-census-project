import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.model import (
    train_model,
    inference,
    performance_on_categorical_slice,
)
from ml.data import process_data


def test_train_model_returns_model():
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])

    model = train_model(X, y)

    assert isinstance(model, LogisticRegression)


def test_inference_output_shape():
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])
    model = train_model(X, y)

    preds = inference(model, X)

    assert len(preds) == len(X)


def test_process_data_output_shapes():
    df = pd.DataFrame({
        "age": [25, 40],
        "workclass": ["Private", "Self-emp"],
        "salary": ["<=50K", ">50K"]
    })

    X, y, encoder, lb = process_data(
        df,
        categorical_features=["workclass"],
        label="salary",
        training=True
    )

    assert X.shape[0] == df.shape[0]
    assert len(y) == df.shape[0]


def test_performance_on_categorical_slice_returns_dict():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    X = pd.DataFrame({
        "workclass": ["Private", "Private", "Self-emp", "Self-emp"]
    })

    results = performance_on_categorical_slice(
        y_true,
        y_pred,
        X,
        "workclass"
    )

    assert isinstance(results, dict)
    assert "Private" in results
    assert "precision" in results["Private"]
   