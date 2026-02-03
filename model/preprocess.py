import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def load_data(path="data/telco_churn_cleaned.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y

def build_preprocessor(X: pd.DataFrame):
    # Identify column types dynamically (safer than hardcoding)
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=25, stratify=y
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Fit-transform once to confirm it works
    Xt = preprocessor.fit_transform(X_train)
    print("Transformed train shape:", Xt.shape)