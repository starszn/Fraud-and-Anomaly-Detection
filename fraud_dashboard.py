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
import os

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
# ULTRA GLASS SHADCN CSS + CENTERED LAYOUT
# ---------------------------------------------------------
st.markdown("""
<style>

/* CENTERED SHADCN LAYOUT */
.main .block-container {
    max-width: 1320px;
    margin-left: auto;
    margin-right: auto;
    padding: 0 !important;
}

/* Remove Streamlit header padding */
header[data-testid="stHeader"] {
    background: transparent;
    height: 0px;
    padding: 0;
    margin: 0;
}

/* Remove Streamlit internal padding */
[data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"],
[data-testid="column"] {
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Background */
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

/* ULTRA GLASS — SHADCN STYLE */
.glass-card {
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(40px) saturate(180%);
    -webkit-backdrop-filter: blur(40px) saturate(180%);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow:
        0 0 25px rgba(255, 0, 255, 0.35),
        0 4px 30px rgba(0, 0, 0, 0.45);
    padding: 20px;
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* Equal-height flexbox row */
.equal-row {
    display: flex;
    gap: 16px;
}

.equal-row > div {
    flex: 1;
    display: flex;
}

/* HEADERS */
.section-header {
    font-size: 24px;
    font-weight: 600;
    color: #F5E8FF;
    margin-bottom: 10px;
}

/* METRICS */
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

/* DataFrame text */
[data-testid="stDataFrame"] {
    color: #EDE6FF !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(5, 5, 20, 0.85);
    backdrop-filter: blur(30px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown(
    "<h2 style='color:#E0D8FF; margin-bottom: 1rem;'>📊 Dashboard</h2>",
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
    "<h1 style='text-align:center; color:#F5E8FF; margin-bottom: 24px;'>🔍 Fraud Detection Dashboard</h1>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# CUSTOM PLOTLY THEME (GLASS STYLE)
# ---------------------------------------------------------
def glass_histogram(df):
    fig = px.histogram(
        df,
        x="anomaly_score",
        nbins=50,
        color_discrete_sequence=["#FF4FFB"]
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#EDE6FF",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor="rgba(255,255,255,0.25)",
            tickfont=dict(color="#CBB4FF")
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor="rgba(255,255,255,0.25)",
            tickfont=dict(color="#CBB4FF")
        ),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig

# ---------------------------------------------------------
# SYSTEM OVERVIEW
# ---------------------------------------------------------
if page == "System Overview":

    st.markdown("<div class='section-header'>📊 Fraud Metrics</div>", unsafe_allow_html=True)

    # 4 equal-width metric cards
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
                card_html = f"""
                <div class='glass-card'>
                    <div class='metric-title'>{title}</div>
                    <div class='metric-value'>{value}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


    # Chart + side metrics
    left, right = st.columns([2.2, 1])

    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📈 Anomaly Score Distribution</div>", unsafe_allow_html=True)
        fig = glass_histogram(transactions_fe)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        side_metrics = {
            "High-Risk Customers": transactions_fe.groupby("customer_id")["anomaly_score"].mean().gt(0.7).sum(),
            "Flagged Transactions": transactions_fe["predicted_anomaly"].sum(),
            "Model Version": "v1.0.0"
        }
        for title, value in side_metrics.items():
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-title'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # High-risk customers
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
                st.dataframe(cust_df.head(20), use_container_width=True)
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
        col1, col2, col3 = st.columns(3)

        details = [
            ("Total Transactions", len(cust_df)),
            ("Avg Anomaly Score", round(cust_df["anomaly_score"].mean(), 4)),
            ("Detected Anomalies", cust_df["predicted_anomaly"].sum())
        ]

        for col, (title, value) in zip([col1, col2, col3], details):
            with col:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-title'>{title}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

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
