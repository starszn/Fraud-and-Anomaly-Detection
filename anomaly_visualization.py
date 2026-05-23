# ---------------------------------------------------------
# 1. Anomaly Score Distribution
# ---------------------------------------------------------
import pandas as pd
transactions_fe = pd.read_pickle("transactions_fe.pkl")
import matplotlib.pyplot as plt
import seaborn as sns



plt.figure(figsize=(10,5))
sns.histplot(transactions_fe["anomaly_score"], bins=50, kde=True)
plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.show()

# ---------------------------------------------------------
# 2. Top Suspicious Transactions
# ---------------------------------------------------------

top_n = 50
top_cases = transactions_fe.nlargest(top_n, "anomaly_score")

plt.figure(figsize=(12,6))
sns.barplot(
    x=top_cases.index,
    y=top_cases["anomaly_score"],
    hue=top_cases["is_anomaly"]
)
plt.title(f"Top {top_n} Most Suspicious Transactions")
plt.xlabel("Transaction Index")
plt.ylabel("Anomaly Score")
plt.legend(title="Actual Anomaly")
plt.show()

# ---------------------------------------------------------
# 3. Scatter Plot of Amount vs. Anomaly Score
# ---------------------------------------------------------

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=transactions_fe,
    x="amount",
    y="anomaly_score",
    hue="is_anomaly",
    alpha=0.6
)
plt.title("Amount vs. Anomaly Score")
plt.xlabel("Transaction Amount")
plt.ylabel("Anomaly Score")
plt.show()

# ---------------------------------------------------------
# 4. Scatter Plot of Distance vs. Anomaly Score
# ---------------------------------------------------------

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=transactions_fe,
    x="distance_from_last_tx",
    y="anomaly_score",
    hue="is_anomaly",
    alpha=0.6
)
plt.title("Distance From Last Transaction vs. Anomaly Score")
plt.xlabel("Distance (km)")
plt.ylabel("Anomaly Score")
plt.show()

# ---------------------------------------------------------
# 5. Time Series of Anomaly Scores for a Single Customer
# ---------------------------------------------------------

cust = transactions_fe[transactions_fe.customer_id == "CUST_00010"]

plt.figure(figsize=(12,6))
plt.plot(cust["timestamp"], cust["anomaly_score"])
plt.scatter(cust["timestamp"], cust["anomaly_score"], c=cust["is_anomaly"])
plt.title("Anomaly Score Over Time for Customer CUST_00010")
plt.xlabel("Time")
plt.ylabel("Anomaly Score")
plt.show()

# ---------------------------------------------------------
# 6. Customer-Level Average Risk
# ---------------------------------------------------------

cust_risk = (
    transactions_fe.groupby("customer_id")["anomaly_score"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(10,6))
sns.barplot(x=cust_risk.values, y=cust_risk.index)
plt.title("Top 20 Highest-Risk Customers (Average Anomaly Score)")
plt.xlabel("Average Anomaly Score")
plt.ylabel("Customer ID")
plt.show()

# ---------------------------------------------------------
# 7. Feature Relationship Clusters
# ---------------------------------------------------------

sample = transactions_fe.sample(2000)

sns.pairplot(
    sample,
    vars=["amount_zscore", "tx_count_1h", "distance_from_last_tx", "anomaly_score"],
    hue="predicted_anomaly",
    plot_kws={"alpha":0.5}
)
plt.show()

# ---------------------------------------------------------
# 8. System-Wide Anomaly Timeline
# ---------------------------------------------------------

df_time = (
    transactions_fe
    .set_index("timestamp")
    .resample("1H")["anomaly_score"]
    .mean()
)

plt.figure(figsize=(14,6))
plt.plot(df_time.index, df_time.values)
plt.title("System-Wide Average Anomaly Score Over Time")
plt.xlabel("Time")
plt.ylabel("Avg Anomaly Score")
plt.show()
