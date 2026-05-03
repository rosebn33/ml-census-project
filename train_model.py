import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    save_model,
    performance_on_categorical_slice,
)

# Load data
data = pd.read_csv("data/census.csv")

# Define categorical features
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

# Split data
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Process training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train model
model = train_model(X_train, y_train)

# Inference
y_pred = inference(model, X_test)

# Overall performance
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
fbeta = fbeta_score(y_test, y_pred, beta=1)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {fbeta:.4f}")

# Save model and encoders
save_model(model, encoder, lb)

# Slice performance
with open("slice_output.txt", "w") as f:
    for feature in cat_features:
        results = performance_on_categorical_slice(
            y_test,
            y_pred,
            test,
            feature,
        )

        for category, metrics in results.items():
            f.write(
                f"{feature} = {category} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1: {metrics['fbeta']:.4f}\n"
            )
