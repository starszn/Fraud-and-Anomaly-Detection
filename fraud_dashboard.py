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
# CSS (Glass + Hover Lift)
# ---------------------------------------------------------
st.markdown("""
<style>

.main .block-container {
    max-width: 1320px;
    margin-left: auto;
    margin-right: auto;
    padding: 0 !important;
}

body {
    background: radial-gradient(circle at top left,
        #050510 0%,
        #09071a 35%,
        #120a2a 65%,
        #1e0f3f 100%);
    background-attachment: fixed;
    color: #EDE6FF;
    font-family: 'Inter', sans-serif;
}

.glass-card {
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(40px) saturate(180%);
    -webkit-backdrop-filter: blur(40px) saturate(180%);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 20px;
    width: 100%;
    display: flex;
    flex-direction: column;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    overflow: hidden;
}

.glass-card:hover {
    transform: translateY(-6px);
    box-shadow:
        0 0 35px rgba(255, 0, 255, 0.45),
        0 6px 40px rgba(0, 0, 0, 0.55);
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

    # Metric cards
    with st.container():
        st.markdown("<div class='equal-row'>", unsafe_allow_html=True)
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

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # ⭐ ALTAR PIE CHART (INSIDE GLASS CARD)
    # ---------------------------------------------------------
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

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📈 Anomaly Score Distribution</div>", unsafe_allow_html=True)
    st.altair_chart(pie_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SIDE METRICS
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # ⭐ ALTAR TABLE (INSIDE GLASS CARD)
    # ---------------------------------------------------------
    st.markdown("<div class='section-header'>🔥 Highest-Risk Customers</div>", unsafe_allow_html=True)

    risk_df = (
        transactions_fe.groupby("customer_id")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    table_chart = (
        alt.Chart(risk_df.head(20))
        .mark_text(align="left", baseline="middle", dx=5)
        .encode(
            y=alt.Y("customer_id:N", sort="-x", title="Customer ID"),
            x=alt.X("anomaly_score:Q", title="Anomaly Score"),
            text=alt.Text("anomaly_score:Q", format=".4f"),
            color=alt.Color("anomaly_score:Q", scale=alt.Scale(scheme="inferno"))
        )
        .properties(height=400)
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.altair_chart(table_chart, use_container_width=True)
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
                st.success(f"Found {len(cust_df)} transactions.")
                st.dataframe(cust_df.head(20), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

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

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("### Recent Transactions")
        st.dataframe(cust_df.tail(20), use_container_width=True)
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
