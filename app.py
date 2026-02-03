import streamlit as st
import pandas as pd
import io
import numpy as np

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
st.title("Telcom Churn Predictor")


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

# Sidebar: controls and upload
st.sidebar.header("Controls")
st.sidebar.markdown("**Training configuration**")
st.sidebar.write(f"Random State: **{RANDOM_STATE}**")
st.sidebar.write(f"Train file: **{TRAIN_DATA_PATH}**")
retrain = st.sidebar.button("Re-train models (clear cache)")

st.sidebar.markdown("---")


st.sidebar.markdown("**Upload / Sample Data**")
uploaded_file = st.sidebar.file_uploader("Upload CSV (same features as training)", type=["csv"])


if retrain:
    st.cache_resource.clear()
    st.success("Cache cleared. Models will retrain on next run.")
    st.experimental_rerun()

# Train / load models with a spinner to communicate progress
with st.spinner("Training / loading models. This may take a moment..."):
    trained_models, metrics_df = train_all_models_cached()

# Update the selectbox now that models are available
selected_model_name = st.sidebar.selectbox("Choose model", options=sorted(trained_models.keys()))

# Top-level header and summary
st.markdown("# 🚀 Telco Customer Churn Predictor")
st.markdown("Predict customer churn using a set of trained ML models. Use the sidebar to choose a model and upload data.")

# Highlight the best model
best_idx = metrics_df["AUC"].idxmax()
best_row = metrics_df.loc[best_idx]
best_name = best_row["ML Model Name"]

m1, m2, m3 = st.columns(3)
m1.metric("Top model", best_name)
m2.metric("Top AUC", f"{best_row['AUC']:.4f}")
m3.metric("Top Accuracy", f"{best_row['Accuracy']:.4f}")

st.divider()

# Metrics table and charts
left, right = st.columns([1.6, 1])
with left:
    st.subheader("Model performance (training split)")
    show_df = metrics_df.copy()
    for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        show_df[c] = show_df[c].round(4)
    show_df = show_df.sort_values("AUC", ascending=False)

    st.dataframe(show_df.set_index("ML Model Name"), use_container_width=True)

with right:
    st.subheader("AUC comparison")
    auc_df = metrics_df.set_index("ML Model Name")["AUC"].sort_values(ascending=True)
    st.bar_chart(auc_df)

st.divider()

# Upload instructions
st.subheader("Prediction inputs")
if uploaded_file is None:
    st.info("Upload a CSV in the sidebar (use the sample CSV as guide).\nThe file should contain the same feature columns used for training. If it includes a 'Churn' column, the app will evaluate prediction quality.")
    st.stop()

# Read uploaded CSV
df = pd.read_csv(uploaded_file)

with st.expander("Uploaded data preview", expanded=True):
    st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)

# Prepare input features
if "Churn" in df.columns:
    X_in = df.drop(columns=["Churn"]) 
    y_true = df["Churn"]
    has_labels = True
else:
    X_in = df.copy()
    has_labels = False

# Perform predictions using selected model
model = trained_models[selected_model_name]
y_prob = model.predict_proba(X_in)[:, 1]
y_pred = model.predict(X_in)

out_df = X_in.copy()
out_df["Churn_Probability"] = np.round(y_prob, 4)
out_df["Predicted_Churn"] = y_pred

# Show sample of predictions and offer download
st.subheader("Predictions")
with st.expander("Prediction sample", expanded=True):
    st.dataframe(out_df.head(20), use_container_width=True)

# Download predictions as CSV
csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

# If labels are present, show evaluation metrics and confusion matrix
if has_labels:
    st.subheader("Evaluation on provided labels")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"])

    with st.expander("Confusion matrix", expanded=True):
        # styled df provides a heatmap-like background
        st.write(cm_df.style.background_gradient(cmap="Blues"))

    with st.expander("Classification report", expanded=True):
        report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
        report_df["precision"] = report_df["precision"].round(4)
        report_df["recall"] = report_df["recall"].round(4)
        report_df["f1-score"] = report_df["f1-score"].round(4)
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(report_df, use_container_width=True)

else:
    st.info("No ground-truth labels found in uploaded data. Showing prediction insights.")
    st.subheader("Prediction distribution")
    dist = pd.Series(y_pred).value_counts().sort_index()
    st.bar_chart(dist)
    st.subheader("Average churn probability")
    st.metric("Average probability", f"{float(y_prob.mean()):.4f}")

st.markdown("---")
st.caption("Tip: Use the model selection in the sidebar to compare predictions and metrics.")
