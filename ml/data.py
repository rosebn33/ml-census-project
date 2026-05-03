from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



def process_data(
    data,
    categorical_features=None,
    label=None,
    training=True,
    encoder=None,
    lb=None,
):
    """
    Process the data used in the machine learning pipeline.
    """
    if categorical_features is None:
        categorical_features = []

    if label is not None:
        y = data[label]
        X = data.drop([label], axis=1)
    else:
        y = None
        X = data

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_features = list(set(X.columns) - set(categorical_features))
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numerical", numerical_transformer, numerical_features),
        ]
    )

    if training:
        X_processed = preprocessor.fit_transform(X)
        lb = LabelBinarizer()
        y = lb.fit_transform(y.values).ravel()
        return X_processed, y, preprocessor, lb

    X_processed = encoder.transform(X)
    y = lb.transform(y.values).ravel() if y is not None else None
    return X_processed, y, encoder, lb
