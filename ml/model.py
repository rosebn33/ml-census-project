import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score



def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def inference(model, X):
    return model.predict(X)

def save_model(model, encoder, lb, path="model/model.joblib"):
    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "label_binarizer": lb,
        },
        path,
    )

def load_model(path="model/model.joblib"):
    objects = joblib.load(path)
    return (
        objects["model"],
        objects["encoder"],
        objects["label_binarizer"],
    )

def performance_on_categorical_slice(y_true, y_pred, X, categorical_feature):
    results = {}

    for category in np.unique(X[categorical_feature]):
        idx = X[categorical_feature] == category

        precision = precision_score(y_true[idx], y_pred[idx], zero_division=1)
        recall = recall_score(y_true[idx], y_pred[idx], zero_division=1)
        fbeta = fbeta_score(y_true[idx], y_pred[idx], beta=1, zero_division=1)

        results[category] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        }

    return results
