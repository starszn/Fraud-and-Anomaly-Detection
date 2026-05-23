import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    return pd.read_pickle(os.path.join(BASE_DIR, "transactions_fe.pkl"))

transactions_fe = load_data()
iso = joblib.load(os.path.join(BASE_DIR, "isolation_forest_model.pkl"))

# ---------------------------------------------------------
# ULTRA GLASS SHADCN CSS (EXACT PORT)
# ---------------------------------------------------------
st.markdown("""
<style>

body {
    background: radial-gradient(circle at top left,
        #050510 0%,
        #09071a 35%,
        #120a2a 65%,
        #1e0f3f 100%);
    background-attachment: fixed;
    color: #EDE6FF;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.01em;
}

/* ULTRA GLASS — EXACT SHADCN STYLE */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(40px) saturate(180%);
    -webkit-backdrop-filter: blur(40px) saturate(180%);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow:
        0 0 0 1px rgba(255, 255, 255, 0.08),
        0 4px 30px rgba(0, 0, 0, 0.45),
        0 0 25px rgba(255, 0, 255, 0.25);
    padding: 20px 24px;
    transition: all 0.25s ease;
}

.glass-card:hover {
    transform: translateY(-3px);
    box-shadow:
        0 0 0 1px rgba(255, 255, 255, 0.2),
        0 6px 40px rgba(0, 0, 0, 0.55),
        0 0 35px rgba(255, 0, 255, 0.35);
}

/* HEADERS */
.section-header {
    font-size: 26px;
    font-weight: 600;
    color: #F5E8FF;
    margin-bottom: 14px;
}

/* METRICS */
.metric-title {
    font-size: 14px;
    color: #CBB4FF;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #FF4FFB;
}

/* TABLES */
[data-testid="stDataFrame"] {
    color: #EDE6FF !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown(
    "<h2 style='color:#E0D8FF;'>📊 Dashboard</h2>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    ["System Overview", "Customer Search", "Customer Details", "Risk Ranking"],
    index=0
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#F5E8FF; margin-bottom: 30px;'>🔍 Fraud Detection Dashboard</h1>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# SYSTEM OVERVIEW
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

    # --- Tier 2: Chart + Activity Cards ---
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
            font_color="#EDE6FF"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
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

    # --- Tier 3: High-Risk Customers ---
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
