import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
else:
    BASE_DIR = os.getcwd()

@st.cache_data
def load_data():
    return pd.read_pickle(os.path.join(BASE_DIR, "transactions_fe.pkl"))

transactions_fe = load_data()
iso = joblib.load(os.path.join(BASE_DIR, "isolation_forest_model.pkl"))

# ---------------------------------------------------------
# CSS (Ultra Glass + Gradient Background)
# ---------------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0a021a 0%,
        #1a0533 20%,
        #2b0a55 40%,
        #3d0f77 60%,
        #4f14a0 80%,
        #5f1ac7 100%
    );
    background-attachment: fixed;
    color: #EDE6FF;
    font-family: 'Inter', sans-serif;
}

/* ULTRA GLASS — Metric Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(80px) saturate(300%);
    -webkit-backdrop-filter: blur(80px) saturate(300%);
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 22px;
    width: 100%;
    display: flex;
    flex-direction: column;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    overflow: hidden;
}

.glass-card:hover {
    transform: translateY(-8px);
    box-shadow:
        0 0 55px rgba(255, 0, 255, 0.55),
        0 12px 60px rgba(0, 0, 0, 0.65);
}

/* ULTRA GLASS — Altair Chart Containers */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed) {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(80px) saturate(300%);
    -webkit-backdrop-filter: blur(80px) saturate(300%);
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 22px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    overflow: hidden;
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed):hover {
    transform: translateY(-8px);
    box-shadow:
        0 0 55px rgba(255, 0, 255, 0.55),
        0 12px 60px rgba(0, 0, 0, 0.65);
}

/* ULTRA GLASS — DataFrame Containers */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.stDataFrame) {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(80px) saturate(300%);
    -webkit-backdrop-filter: blur(80px) saturate(300%);
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 22px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    overflow: hidden;
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.stDataFrame):hover {
    transform: translateY(-8px);
    box-shadow:
        0 0 55px rgba(255, 0, 255, 0.55),
        0 12px 60px rgba(0, 0, 0, 0.65);
}

.equal-row {
    display: flex;
    gap: 16px;
}

.equal-row > div {
    flex: 1;
    display: flex;
}

.section-header {
    font-size: 24px;
    font-weight: 600;
    color: #F5E8FF;
    margin-bottom: 10px;
}

.metric-title {
    font-size: 13px;
    color: #CBB4FF;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #FF4FFB;
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
    ["System Overview", "Customer Search", "Customer Details", "Risk Ranking"]
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#F5E8FF;'>🔍 Fraud Detection Dashboard</h1>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# SYSTEM OVERVIEW
# ---------------------------------------------------------
if page == "System Overview":

    st.markdown("<div class='section-header'>📊 Fraud Metrics</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("Total Transactions", f"{len(transactions_fe):,}"),
        ("Unique Customers", f"{transactions_fe['customer_id'].nunique():,}"),
        ("Detected Anomalies", f"{transactions_fe['predicted_anomaly'].sum():,}"),
        ("Avg Anomaly Score", f"{round(transactions_fe['anomaly_score'].mean(), 4)}")
    ]

    for col, (title, value) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f"""
                <div class='glass-card'>
                    <div class='metric-title'>{title}</div>
                    <div class='metric-value'>{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # PIE CHART
    pie_data = transactions_fe.copy()
    pie_data["bucket"] = pd.cut(
        pie_data["anomaly_score"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    )

    pie_chart = (
        alt.Chart(pie_data)
        .mark_arc(outerRadius=120)
        .encode(
            theta="count():Q",
            color=alt.Color("bucket:N", scale=alt.Scale(scheme="magma")),
            tooltip=["bucket:N", "count():Q"]
        )
        .properties(height=400)
    )

    with st.container():
        st.markdown("<div class='section-header'>📈 Anomaly Score Distribution</div>", unsafe_allow_html=True)
        st.altair_chart(pie_chart, use_container_width=True)

    # SIDE METRICS
    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)

    side_metrics = [
        ("High-Risk Customers", transactions_fe.groupby("customer_id")["anomaly_score"].mean().gt(0.7).sum()),
        ("Flagged Transactions", transactions_fe["predicted_anomaly"].sum()),
        ("Model Version", "v1.0.0")
    ]

    for col, (title, value) in zip([colA, colB, colC], side_metrics):
        with col:
            st.markdown(
                f"""
                <div class='glass-card'>
                    <div class='metric-title'>{title}</div>
                    <div class='metric-value'>{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # DATAFRAME
    st.markdown("<br>", unsafe_allow_html=True)

    risk_df = (
        transactions_fe.groupby("customer_id")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    with st.container():
        st.markdown("<div class='section-header'>🔥 Highest-Risk Customers</div>", unsafe_allow_html=True)
        st.dataframe(risk_df.head(20), use_container_width=True)

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

            with st.container():
                if cust_df.empty:
                    st.warning("Customer not found.")
                else:
                    st.success(f"Found {len(cust_df)} transactions.")
                    st.dataframe(cust_df.head(20), use_container_width=True)

        except:
            st.error("Please enter a valid numeric ID.")

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
        col1, col2, col3 = st.columns(3)

        details = [
            ("Total Transactions", len(cust_df)),
            ("Avg Anomaly Score", round(cust_df["anomaly_score"].mean(), 4)),
            ("Detected Anomalies", cust_df["predicted_anomaly"].sum())
        ]

        for col, (title, value) in zip([col1, col2, col3], details):
            with col:
                st.markdown(
                    f"""
                    <div class='glass-card'>
                        <div class='metric-title'>{title}</div>
                        <div class='metric-value'>{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with st.container():
            st.write("### Recent Transactions")
            st.dataframe(cust_df.tail(20), use_container_width=True)

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

    with st.container():
        st.dataframe(risk_df.head(20), use_container_width=True)
