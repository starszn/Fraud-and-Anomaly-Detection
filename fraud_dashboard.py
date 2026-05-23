import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# ---------------------------------------------------------
# MUST BE FIRST STREAMLIT COMMAND
# ---------------------------------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ---------------------------------------------------------
# Load Data & Model
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    return pd.read_pickle(os.path.join(BASE_DIR, "transactions_fe.pkl"))

transactions_fe = load_data()
iso = joblib.load(os.path.join(BASE_DIR, "isolation_forest_model.pkl"))

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <h2 style='color:#E0E0FF;'>📊 Dashboard</h2>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    ["System Overview", "Customer Search", "Customer Details", "Risk Ranking"],
    index=0
)

# ---------------------------------------------------------
# THEME: Navy‑Purple Glassmorphism + Magenta Accent
# ---------------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0a0f2d, #1b0f3b, #2a0f4f);
    background-attachment: fixed;
}

/* Frosted Glass Panels */
.glass-card {
    background: rgba(10, 10, 25, 0.18);
    backdrop-filter: blur(28px) saturate(180%);
    -webkit-backdrop-filter: blur(28px) saturate(180%);
    border-radius: 18px;
    padding: 20px 24px;
    border: 1px solid rgba(255, 0, 255, 0.15);
    box-shadow: 0 8px 32px rgba(255, 0, 255, 0.12);
    margin-bottom: 22px;
    color: #E8E0FF;
}

/* Section Headers */
.section-header {
    font-size: 28px;
    font-weight: 600;
    color: #F0D0FF;
    margin-bottom: 14px;
}

/* Metric Titles */
.metric-title {
    font-size: 16px;
    color: #D8C0FF;
}

/* Metric Values */
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #FF4FFB;
}

/* Table Styling */
[data-testid="stDataFrame"] {
    color: #E8E0FF !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#F0D0FF; margin-bottom: 30px;'>🔍 Fraud Detection Dashboard</h1>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# SYSTEM OVERVIEW (Balanced Compact Layout)
# ---------------------------------------------------------
if page == "System Overview":

    st.markdown("<div class='section-header'>📊 Fraud Metrics</div>", unsafe_allow_html=True)

    # --- Tier 1: Metrics ---
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Total Transactions</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(transactions_fe):,}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Unique Customers</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{transactions_fe['customer_id'].nunique():,}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Detected Anomalies</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{transactions_fe['predicted_anomaly'].sum():,}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tier 2: Main Chart + Activity Cards ---
    left, right = st.columns([2.2, 1])

    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📈 Anomaly Score Distribution</div>", unsafe_allow_html=True)

        fig = px.histogram(
            transactions_fe,
            x="anomaly_score",
            nbins=50,
            color_discrete_sequence=["#FF4FFB"]
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8E0FF"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        # Activity Cards
        metrics = {
            "High-Risk Customers": transactions_fe.groupby("customer_id")["anomaly_score"].mean().gt(0.7).sum(),
            "Flagged Transactions": transactions_fe["predicted_anomaly"].sum(),
            "Avg Anomaly Score": round(transactions_fe["anomaly_score"].mean(), 4),
            "Model Version": "v1.0.0"
        }

        for title, value in metrics.items():
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-title'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- Tier 3: High-Risk Customer List ---
    st.markdown("<div class='section-header'>🔥 Highest-Risk Customers</div>", unsafe_allow_html=True)

    risk_df = (
        transactions_fe.groupby("customer_id")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.dataframe(risk_df.head(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# CUSTOMER SEARCH
# ---------------------------------------------------------
elif page == "Customer Search":
    st.markdown("<div class='section-header'>🔎 Customer Search</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    customer_id = st.text_input("Enter Customer ID:")
    st.markdown("</div>", unsafe_allow_html=True)

    if customer_id:
        try:
            customer_id = int(customer_id)
            cust_df = transactions_fe[transactions_fe["customer_id"] == customer_id]

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            if cust_df.empty:
                st.warning("Customer not found.")
            else:
                st.success(f"Found {len(cust_df)} transactions for customer {customer_id}.")
                st.dataframe(cust_df.head(20))
            st.markdown("</div>", unsafe_allow_html=True)

        except:
            st.error("Please enter a valid numeric customer ID.")

# ---------------------------------------------------------
# CUSTOMER DETAILS
# ---------------------------------------------------------
elif page == "Customer Details":
    st.markdown("<div class='section-header'>👤 Customer Details</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    customer_id = st.number_input("Enter Customer ID:", min_value=1, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

    cust_df = transactions_fe[transactions_fe["customer_id"] == customer_id]

    if not cust_df.empty:
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>Total Transactions</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{len(cust_df)}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>Avg Anomaly Score</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{round(cust_df['anomaly_score'].mean(), 4)}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>Detected Anomalies</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{cust_df['predicted_anomaly'].sum()}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("### Recent Transactions")
        st.dataframe(cust_df.tail(20))
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Enter a valid customer ID.")

# ---------------------------------------------------------
# RISK RANKING
# ---------------------------------------------------------
elif page == "Risk Ranking":
    st.markdown("<div class='section-header'>🔥 Highest-Risk Customers</div>", unsafe_allow_html=True)

    risk_df = (
        transactions_fe.groupby("customer_id")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.dataframe(risk_df.head(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
