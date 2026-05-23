import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os

from table_creation import (
    generate_customers,
    generate_normal_transactions,
    inject_anomalies
)

# Base directory of THIS script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_transactions(df):
    df = df.sort_values(["customer_id", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def add_transaction_velocity(df, window_minutes=60):
    df["tx_count_1h"] = 0
    for cust in df["customer_id"].unique():
        cust_df = df[df.customer_id == cust]
        times = cust_df["timestamp"].values
        counts = []
        for i in range(len(times)):
            current_time = times[i]
            window_start = current_time - np.timedelta64(window_minutes, "m")
            count = np.sum((times >= window_start) & (times <= current_time))
            counts.append(count)
        df.loc[cust_df.index, "tx_count_1h"] = counts
    return df

def add_amount_zscore(df):
    df["amount_zscore"] = 0.0
    for cust in df["customer_id"].unique():
        cust_df = df[df.customer_id == cust]
        mean = cust_df["amount"].mean()
        std = cust_df["amount"].std() + 1e-6
        zscores = (cust_df["amount"] - mean) / std
        df.loc[cust_df.index, "amount_zscore"] = zscores
    return df

def add_distance_from_last_tx(df):
    df["distance_from_last_tx"] = 0.0
    for cust in df["customer_id"].unique():
        cust_df = df[df.customer_id == cust].sort_values("timestamp")
        distances = [0]
        prev_lat, prev_lon = cust_df.iloc[0][["lat", "lon"]]
        for i in range(1, len(cust_df)):
            lat, lon = cust_df.iloc[i][["lat", "lon"]]
            dist = geodesic((prev_lat, prev_lon), (lat, lon)).km
            distances.append(dist)
            prev_lat, prev_lon = lat, lon
        df.loc[cust_df.index, "distance_from_last_tx"] = distances
    return df

def add_merchant_novelty(df):
    df["merchant_novelty"] = 0
    for cust in df["customer_id"].unique():
        cust_df = df[df.customer_id == cust].sort_values("timestamp")
        seen_merchants = set()
        novelty = []
        for _, row in cust_df.iterrows():
            merchant = row["merchant_category"]
            novelty.append(0 if merchant in seen_merchants else 1)
            seen_merchants.add(merchant)
        df.loc[cust_df.index, "merchant_novelty"] = novelty
    return df

def add_time_of_day_anomaly(df):
    df["hour"] = df["timestamp"].dt.hour
    df["time_of_day_anomaly"] = df["hour"].apply(lambda h: 1 if (h < 6 or h > 23) else 0)
    return df

def add_device_change(df):
    df["device_change"] = 0
    for cust in df["customer_id"].unique():
        cust_df = df[df.customer_id == cust].sort_values("timestamp")
        prev_device = None
        changes = []
        for _, row in cust_df.iterrows():
            device = row["device_id"]
            if prev_device is None:
                changes.append(0)
            else:
                changes.append(1 if device != prev_device else 0)
            prev_device = device
        df.loc[cust_df.index, "device_change"] = changes
    return df

def build_features(df):
    df = prepare_transactions(df)
    df = add_transaction_velocity(df)
    df = add_amount_zscore(df)
    df = add_distance_from_last_tx(df)
    df = add_merchant_novelty(df)
    df = add_time_of_day_anomaly(df)
    df = add_device_change(df)
    return df

customers = generate_customers(500)
transactions = generate_normal_transactions(customers, 50000)
transactions = inject_anomalies(transactions, customers)

transactions_fe = build_features(transactions)

save_path = os.path.join(BASE_DIR, "transactions_fe.pkl")
print("Saving to:", save_path)
transactions_fe.to_pickle(save_path)
