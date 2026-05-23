import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from geopy.distance import geodesic

# -----------------------------
# 1. Generate Customers
# -----------------------------

def generate_customers(n_customers=500):
    customers = []

    for i in range(n_customers):
        customer_id = f"CUST_{i:05d}"
        age = np.random.randint(18, 80)
        gender = random.choice(["M", "F"])
        income = np.random.normal(70000, 20000)
        account_type = random.choice(["checking", "savings", "business"])

        # Home location (US-based)
        home_lat = np.random.uniform(29.0, 30.0)   # Houston-ish
        home_lon = np.random.uniform(-96.0, -94.0)

        usual_countries = ["USA"]
        risk_score = np.random.uniform(0.1, 0.9)

        customers.append([
            customer_id, age, gender, income, account_type,
            home_lat, home_lon, usual_countries, risk_score
        ])

    return pd.DataFrame(customers, columns=[
        "customer_id", "age", "gender", "income", "account_type",
        "home_lat", "home_lon", "usual_countries", "risk_score"
    ])


# -----------------------------
# 2. Generate Normal Transactions
# -----------------------------

merchant_categories = [
    "groceries", "gas", "restaurants", "clothing", "electronics",
    "travel", "utilities", "entertainment", "atm", "transfer"
]

transaction_types = ["POS", "online", "ATM", "transfer"]
channels = ["mobile_app", "web", "card_swipe"]

def generate_normal_transactions(customers, n_transactions=50000):
    transactions = []

    for _ in range(n_transactions):
        cust = customers.sample(1).iloc[0]
        customer_id = cust["customer_id"]

        # Timestamp
        timestamp = datetime(2024, 1, 1) + timedelta(
            minutes=np.random.randint(0, 365*24*60)
        )

        # Amount based on income
        base_amount = np.random.exponential(cust["income"] / 2000)
        amount = round(min(base_amount, 2000), 2)

        merchant = random.choice(merchant_categories)
        ttype = random.choice(transaction_types)
        channel = random.choice(channels)

        # Location near home
        lat = cust["home_lat"] + np.random.normal(0, 0.05)
        lon = cust["home_lon"] + np.random.normal(0, 0.05)

        device_id = f"DEV_{np.random.randint(1, 2000)}"

        transactions.append([
            f"TX_{len(transactions):08d}", customer_id, timestamp,
            amount, merchant, ttype, lat, lon, device_id, channel, 0
        ])

    return pd.DataFrame(transactions, columns=[
        "transaction_id", "customer_id", "timestamp", "amount",
        "merchant_category", "transaction_type", "lat", "lon",
        "device_id", "channel", "is_anomaly"
    ])


# -----------------------------
# 3. Inject Anomalies
# -----------------------------

def inject_anomalies(df, customers, anomaly_rate=0.005):
    n_anomalies = int(len(df) * anomaly_rate)
    anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice([
            "location_jump", "high_velocity", "amount_spike",
            "new_merchant", "new_device", "suspicious_transfer"
        ])

        row = df.loc[idx]
        cust = customers[customers.customer_id == row.customer_id].iloc[0]

        # 1. Location jump
        if anomaly_type == "location_jump":
            df.at[idx, "lat"] = cust["home_lat"] + np.random.uniform(5, 20)
            df.at[idx, "lon"] = cust["home_lon"] + np.random.uniform(5, 20)

        # 2. High velocity (simulate by timestamp shift)
        elif anomaly_type == "high_velocity":
            df.at[idx, "timestamp"] += timedelta(minutes=np.random.randint(1, 3))

        # 3. Amount spike
        elif anomaly_type == "amount_spike":
            df.at[idx, "amount"] *= np.random.uniform(5, 15)

        # 4. New merchant category
        elif anomaly_type == "new_merchant":
            df.at[idx, "merchant_category"] = "luxury_goods"

        # 5. New device
        elif anomaly_type == "new_device":
            df.at[idx, "device_id"] = f"NEW_DEV_{np.random.randint(10000, 99999)}"

        # 6. Suspicious transfer
        elif anomaly_type == "suspicious_transfer":
            df.at[idx, "transaction_type"] = "transfer"
            df.at[idx, "amount"] = np.random.uniform(2000, 10000)

        df.at[idx, "is_anomaly"] = 1

    return df


# -----------------------------
# 4. Run the Generator
# -----------------------------

customers = generate_customers(500)
transactions = generate_normal_transactions(customers, 50000)
transactions = inject_anomalies(transactions, customers)

print("Customers:", customers.shape)
print("Transactions:", transactions.shape)

transactions.head()
