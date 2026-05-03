import pandas as pd


columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "salary"
]

df = pd.read_csv(
    "data/adult.data",
    header=None,
    names=columns,
    skipinitialspace=True
)

df.to_csv("data/census.csv", index=False)

print(" census.csv created successfully")
