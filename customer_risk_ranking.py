import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Load the scored dataset
# ---------------------------------------------------------

transactions_fe = pd.read_pickle("transactions_fe.pkl")

# ---------------------------------------------------------
# 2. Compute customer-level risk
# ---------------------------------------------------------

cust_risk = (
    transactions_fe.groupby("customer_id")["anomaly_score"]
    .mean()
    .sort_values(ascending=False)
)

print("Top 20 Highest-Risk Customers:")
print(cust_risk.head(20))

# ---------------------------------------------------------
# 3. Visualize top-risk customers
# ---------------------------------------------------------

top_n = 20
top_customers = cust_risk.head(top_n)

plt.figure(figsize=(10,6))
sns.barplot(x=top_customers.values, y=top_customers.index)
plt.title(f"Top {top_n} Highest-Risk Customers (Average Anomaly Score)")
plt.xlabel("Average Anomaly Score")
plt.ylabel("Customer ID")
plt.show()

# ---------------------------------------------------------
# 4. Optional: Assign risk tiers
# ---------------------------------------------------------

quantiles = cust_risk.quantile([0.8, 0.95])

def risk_tier(score):
    if score >= quantiles[0.95]:
        return "High Risk"
    elif score >= quantiles[0.8]:
        return "Medium Risk"
    else:
        return "Low Risk"

transactions_fe["risk_tier"] = transactions_fe["customer_id"].map(
    cust_risk.apply(risk_tier)
)

print("\nRisk tier distribution:")
print(transactions_fe["risk_tier"].value_counts())
