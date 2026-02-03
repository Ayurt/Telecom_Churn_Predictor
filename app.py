import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from model.preprocess import build_preprocessor
from model.metrics import compute_classification_metrics


# -------------------- CONFIG --------------------
RANDOM_STATE = 25
TRAIN_DATA_PATH = "data/telco_churn_cleaned.csv"


st.set_page_config(page_title="Telco Churn Classifier", layout="wide")
st.title("Telco Customer Churn Prediction")


# -------------------- DATA LOADING --------------------
@st.cache_data
def load_training_data(path: str):
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("Training file must contain 'Churn' column.")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


# -------------------- MODEL TRAINING --------------------
@st.cache_resource
def train_all_models_cached():
    X, y = load_training_data(TRAIN_DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor, _, _ = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
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

    trained = {}
    results = []

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", clf)
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        m = compute_classification_metrics(y_test, y_pred, y_proba)

        results.append({
            "ML Model Name": name,
            "Accuracy": m["Accuracy"],
            "AUC": m["AUC"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1": m["F1"],
            "MCC": m["MCC"]
        })

        trained[name] = pipe

    metrics_df = pd.DataFrame(results)
    return trained, metrics_df


# -------------------- UI --------------------
st.sidebar.header("Controls")

with st.sidebar.expander("Training configuration", expanded=True):
    st.write(f"Random State: **{RANDOM_STATE}**")
    st.write(f"Train file: **{TRAIN_DATA_PATH}**")
    retrain = st.button("Re-train models (clear cache)")

if retrain:
    st.cache_resource.clear()
    st.success("Cache cleared. Models will retrain on next run.")

trained_models, metrics_df = train_all_models_cached()

# Metrics display
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Model Performance (Training Split)")
    show_df = metrics_df.copy()
    for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        show_df[c] = show_df[c].round(4)
    show_df = show_df.sort_values("AUC", ascending=False)
    st.dataframe(show_df, use_container_width=True)

with col2:
    st.subheader("Select Model for Predictions")
    selected_model_name = st.selectbox("Model", sorted(trained_models.keys()))
    st.write(f"Selected: **{selected_model_name}**")

st.divider()

# Upload test CSV
st.subheader("Upload Test CSV")
uploaded_file = st.file_uploader("Upload CSV (with same feature columns)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to generate predictions.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Uploaded Data Preview")
st.dataframe(df.head(15), use_container_width=True)

model = trained_models[selected_model_name]

# Prediction logic
st.subheader("Predictions")

if "Churn" in df.columns:
    X_in = df.drop(columns=["Churn"])
    y_true = df["Churn"]
    has_labels = True
else:
    X_in = df.copy()
    has_labels = False

y_pred = model.predict(X_in)
y_prob = model.predict_proba(X_in)[:, 1]

out_df = X_in.copy()
out_df["Predicted_Churn"] = y_pred
out_df["Churn_Probability"] = y_prob.round(4)

st.dataframe(out_df.head(20), use_container_width=True)

if has_labels:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.write(pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    ))

    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))
else:
    st.info("No ground-truth labels (`Churn`) found in uploaded data. Showing prediction insights only.")
    st.subheader("Prediction Distribution")
    st.bar_chart(pd.Series(y_pred).value_counts())
    st.subheader("Average Churn Probability")
    st.write(round(float(y_prob.mean()), 4))
