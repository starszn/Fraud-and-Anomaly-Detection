import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "transactions_fe.pkl")
transactions_fe = pd.read_pickle(data_path)

feature_cols = [
    "amount",
    "tx_count_1h",
    "amount_zscore",
    "distance_from_last_tx",
    "merchant_novelty",
    "time_of_day_anomaly",
    "device_change"
]

X = transactions_fe[feature_cols]
y = transactions_fe["is_anomaly"]

iso = IsolationForest(
    n_estimators=200,
    contamination=0.005,
    random_state=42,
    n_jobs=-1
)

iso.fit(X)

pred_labels = iso.predict(X)
pred_labels = np.where(pred_labels == -1, 1, 0)

anomaly_scores = -iso.score_samples(X)
transactions_fe["anomaly_score"] = anomaly_scores
transactions_fe["predicted_anomaly"] = pred_labels

print("Classification Report:")
print(classification_report(y, pred_labels))

roc = roc_auc_score(y, anomaly_scores)
print("ROC-AUC:", roc)

precision, recall, thresholds = precision_recall_curve(y, anomaly_scores)
pr_auc = auc(recall, precision)
print("PR-AUC:", pr_auc)

save_data_path = os.path.join(BASE_DIR, "transactions_fe.pkl")
save_model_path = os.path.join(BASE_DIR, "isolation_forest_model.pkl")

transactions_fe.to_pickle(save_data_path)
joblib.dump(iso, save_model_path)
