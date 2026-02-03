from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef)

def compute_classification_metrics(y_true, y_pred, y_proba):
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred)
    metrics["Recall"] = recall_score(y_true, y_pred)
    metrics["F1"] = f1_score(y_true, y_pred)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    metrics["AUC"] = roc_auc_score(y_true, y_proba)
    return metrics