import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
import joblib

from preprocess import load_data, build_preprocessor
from metrics import compute_classification_metrics


RANDOM_STATE = 25


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def train_and_evaluate(model_name: str, model, preprocessor, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # AUC needs probabilities; all our chosen models support predict_proba
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    m = compute_classification_metrics(y_test, y_pred, y_proba)

    return pipeline, m


def main():
    # 1) Load cleaned data
    X, y = load_data("data/telco_churn_cleaned.csv")

    # 2) Split once (must be identical for all models)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 3) Build preprocessor once (must be identical for all models)
    preprocessor, _, _ = build_preprocessor(X_train)

    # 4) Define models (reasonable defaults; avoid huge training times)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=None
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss"
        )
    }

    # 5) Train, evaluate, and save
    ensure_dir("model/saved_models")
    results = []
    pipelines = {}

    for name, model in models.items():
        pipeline, m = train_and_evaluate(name, model, preprocessor, X_train, X_test, y_train, y_test)

        row = {
            "ML Model Name": name,
            "Accuracy": m["Accuracy"],
            "AUC": m["AUC"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1": m["F1"],
            "MCC": m["MCC"]
        }
        results.append(row)
        pipelines[name] = pipeline

        # Save each pipeline separately (deployment-friendly)
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(pipeline, f"model/saved_models/{safe_name}.joblib")

        print(f"Done: {name}")

    # 6) Save metrics table for README + Streamlit
    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df[["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    metrics_df.to_csv("model/metrics_table.csv", index=False)

    # Also save all pipelines together (optional convenience)
    joblib.dump(pipelines, "model/saved_models/all_models.joblib")

    print("\nSaved: model/metrics_table.csv")
    print("Saved models in: model/saved_models/")
    print("\nMetrics Table (preview):")
    print(metrics_df.sort_values("AUC", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
