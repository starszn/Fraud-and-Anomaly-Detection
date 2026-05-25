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
# CSS
# ---------------------------------------------------------
st.markdown("""
<style>

/* ── Base background — periwinkle/lavender like the reference ── */
.stApp {
    background: linear-gradient(
        135deg,
        #6a7fd4 0%,
        #8b6fc8 30%,
        #b97acd 60%,
        #d48aaa 100%
    );
    background-attachment: fixed;
    min-height: 100vh;
    color: #1a103a;
    font-family: 'Inter', sans-serif;
    position: relative;
    overflow: hidden;
}

/* ── Orb layer — sits behind everything ── */
.stApp::before,
.stApp::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    z-index: 0;
    pointer-events: none;
}

/* Top-center cyan/white orb */
.stApp::before {
    width: 600px;
    height: 600px;
    top: -180px;
    left: 50%;
    transform: translateX(-50%);
    background: radial-gradient(circle,
        rgba(190, 230, 255, 0.90) 0%,
        rgba(140, 190, 255, 0.60) 40%,
        rgba(100, 140, 230, 0.00) 70%
    );
}

/* Bottom-right coral/pink orb */
.stApp::after {
    width: 700px;
    height: 700px;
    bottom: -200px;
    right: -150px;
    background: radial-gradient(circle,
        rgba(255, 160, 140, 0.85) 0%,
        rgba(255, 120, 160, 0.55) 40%,
        rgba(200, 100, 180, 0.00) 70%
    );
}

/* Extra orbs via a wrapper div we inject below */
.orb-white-left {
    position: fixed;
    width: 500px;
    height: 500px;
    bottom: -100px;
    left: -120px;
    border-radius: 50%;
    background: radial-gradient(circle,
        rgba(220, 230, 255, 0.80) 0%,
        rgba(180, 200, 255, 0.45) 45%,
        rgba(150, 170, 240, 0.00) 70%
    );
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
}

/* Ensure Streamlit content sits above orbs */
.stApp > * {
    position: relative;
    z-index: 1;
}

/* ── Shared liquid-glass mixin ── */
.glass-card,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed),
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.stDataFrame) {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(50px) saturate(200%) brightness(1.10);
    -webkit-backdrop-filter: blur(50px) saturate(200%) brightness(1.10);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow:
        inset 0 1.5px 0 rgba(255, 255, 255, 0.70),
        inset 1px 0 0   rgba(255, 255, 255, 0.35),
        0 8px 32px rgba(100, 80, 180, 0.20),
        0 2px 8px  rgba(0,   0,   0,   0.10);
    padding: 22px;
    width: 100%;
    transition: transform 0.28s ease, box-shadow 0.28s ease;
    overflow: hidden;
}

.glass-card:hover,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed):hover,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.stDataFrame):hover {
    transform: translateY(-8px);
    box-shadow:
        inset 0 1.5px 0 rgba(255, 255, 255, 0.80),
        inset 1px 0 0   rgba(255, 255, 255, 0.40),
        0 0 50px rgba(180, 120, 255, 0.35),
        0 16px 60px rgba(0, 0, 0, 0.18);
}

.glass-card {
    display: flex;
    flex-direction: column;
}

/* ── Typography ── */
.section-header {
    font-size: 22px;
    font-weight: 700;
    color: #2a1060;
    margin-bottom: 12px;
    text-shadow: 0 1px 2px rgba(255,255,255,0.4);
}

.metric-title {
    font-size: 11px;
    color: rgba(60, 30, 120, 0.75);
    margin-bottom: 6px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #3d0f9e;
    text-shadow: 0 2px 12px rgba(100, 50, 200, 0.25);
}

/* ── Streamlit widget text overrides for light bg ── */
h1 {
    color: #1a103a !important;
    text-shadow: 0 2px 8px rgba(255,255,255,0.3);
}

label, .stSelectbox label, .stTextInput label, .stNumberInput label {
    color: #2a1060 !important;
}

</style>

<!-- Bottom-left white orb (can't do 3+ pseudo-elements, so inject as HTML) -->
<div class="orb-white-left"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown(
    "<h2 style='color:#2a1060;'>📊 Dashboard</h2>",
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
    "<h1 style='text-align:center; color:#1a103a;'>🔍 Fraud Detection Dashboard</h1>",
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
