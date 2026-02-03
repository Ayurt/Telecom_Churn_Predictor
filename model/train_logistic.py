import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocess import load_data, build_preprocessor
from metrics import compute_classification_metrics

X,y = load_data("data/telco_churn_cleaned.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 25, stratify = y)

preprocessor, _, _ = build_preprocessor(X_train)

log_reg = LogisticRegression(max_iter = 1000, solver = "lbfgs")

pipeline = Pipeline(steps = [("preprocess", preprocessor), ("model", log_reg)])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

metrics = compute_classification_metrics(y_test, y_pred, y_proba)

print("\nLogistic Regression Metrics: ")
for k,v in metrics.items():
    print(f"{k}: {v:.4f}")