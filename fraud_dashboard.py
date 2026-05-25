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
# CSS (Liquid Glass — iOS 26 / reference style)
# ---------------------------------------------------------
st.markdown("""
<style>

/* ── Background ── */
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

/* ── Shared liquid-glass mixin ── */
.glass-card,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed),
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.stDataFrame) {
    /* Near-invisible fill so the gradient bleeds through */
    background: rgba(255, 255, 255, 0.06);

    /* Heavy blur + saturation boost = frosted-glass depth */
    backdrop-filter: blur(60px) saturate(220%) brightness(1.08);
    -webkit-backdrop-filter: blur(60px) saturate(220%) brightness(1.08);

    /* Soft bright border — thicker top/left for light-source illusion */
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.30);
    box-shadow:
        /* inner top-left highlight */
        inset 0 1px 0 rgba(255, 255, 255, 0.45),
        inset 1px 0 0 rgba(255, 255, 255, 0.20),
        /* outer ambient glow */
        0 8px 32px rgba(0, 0, 0, 0.35),
        0 2px 8px  rgba(0, 0, 0, 0.20);

    padding: 22px;
    width: 100%;
    transition: transform 0.28s ease, box-shadow 0.28s ease;
    overflow: hidden;
}

/* ── Hover lift ── */
.glass-card:hover,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed):hover,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.stDataFrame):hover {
    transform: translateY(-8px);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.55),
        inset 1px 0 0 rgba(255, 255, 255, 0.25),
        0 0 55px rgba(180, 80, 255, 0.50),
        0 16px 60px rgba(0, 0, 0, 0.55);
}

/* ── metric card flex layout ── */
.glass-card {
    display: flex;
    flex-direction: column;
}

.section-header {
    font-size: 24px;
    font-weight: 600;
    color: #F5E8FF;
    margin-bottom: 10px;
}

.metric-title {
    font-size: 13px;
    color: rgba(220, 200, 255, 0.80);
    margin-bottom: 6px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #FF4FFB;
    text-shadow: 0 0 20px rgba(255, 79, 251, 0.45);
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
