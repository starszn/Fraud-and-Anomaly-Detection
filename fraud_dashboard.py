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
# Helper — Risk Badge
# ---------------------------------------------------------
def risk_badge(score):
    if score >= 0.7:
        return "<span class='risk-badge risk-high'>High</span>"
    elif score >= 0.4:
        return "<span class='risk-badge risk-medium'>Medium</span>"
    else:
        return "<span class='risk-badge risk-low'>Low</span>"

# ---------------------------------------------------------
# Helper — render a DataFrame as a glass-friendly HTML table
# ---------------------------------------------------------
def glass_table(df: pd.DataFrame):
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{val}</td>" for val in row)
        rows += f"<tr>{cells}</tr>"
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    html = f"""
    <div class="glass-card" style="overflow-x:auto; padding:0;">
        <table class="glass-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------
st.markdown("""
<style>

/* ── Smooth font rendering ── */
* {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ── Background on html only — never scrolls ── */
html {
    background: linear-gradient(145deg,
        #e0e8ff 0%,
        #ead6f8 22%,
        #f8d6ee 44%,
        #eeddf8 66%,
        #d8e6ff 88%,
        #ccd8ff 100%
    ) !important;
    background-attachment: fixed !important;
}

body {
    background: transparent !important;
}

/* ── Transparent Streamlit wrappers ── */
.stApp {
    background: transparent !important;
}

[data-testid="stAppViewContainer"] {
    background: transparent !important;
    overflow-y: auto !important;
    height: 100vh !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

section.main > div,
.block-container {
    background: transparent !important;
}

/* ── Risk Badges ── */
.risk-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: white;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
.risk-high   { background: linear-gradient(135deg, #ff4e88, #d6004a); }
.risk-medium { background: linear-gradient(135deg, #ffb347, #ff7b00); }
.risk-low    { background: linear-gradient(135deg, #4cd964, #1fae4b); }

/* ── Blobs ── */
.blob {
    position: fixed;
    pointer-events: none;
    z-index: 0;
}

.blob-bl {
    width: 560px;
    height: 400px;
    bottom: -90px;
    left: -110px;
    border-radius: 52% 48% 42% 58% / 62% 38% 62% 38%;
    background: radial-gradient(ellipse at 40% 36%,
        rgba(190, 208, 255, 1.00)  0%,
        rgba(168, 188, 255, 0.96) 18%,
        rgba(148, 168, 250, 0.84) 35%,
        rgba(125, 148, 240, 0.58) 55%,
        rgba(102, 125, 228, 0.22) 75%,
        rgba( 80, 102, 215, 0.00) 90%
    );
}

.blob-br {
    width: 500px;
    height: 380px;
    bottom: -70px;
    right: -90px;
    border-radius: 42% 58% 52% 48% / 52% 42% 58% 48%;
    background: radial-gradient(ellipse at 38% 34%,
        rgba(205, 195, 255, 1.00)  0%,
        rgba(182, 172, 252, 0.96) 18%,
        rgba(162, 152, 245, 0.84) 35%,
        rgba(140, 130, 235, 0.58) 55%,
        rgba(118, 108, 222, 0.22) 75%,
        rgba( 96,  88, 208, 0.00) 90%
    );
}

.blob-center {
    width: 560px;
    height: 560px;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border-radius: 50%;
    background: radial-gradient(ellipse at 36% 30%,
        rgba(255, 255, 255, 1.00)  0%,
        rgba(255, 250, 255, 0.98)  8%,
        rgba(252, 238, 255, 0.92) 20%,
        rgba(244, 222, 255, 0.80) 34%,
        rgba(232, 208, 255, 0.60) 50%,
        rgba(218, 195, 255, 0.35) 66%,
        rgba(200, 180, 255, 0.12) 82%,
        rgba(180, 165, 255, 0.00) 95%
    );
}

.blob-pink-l {
    width: 200px;
    height: 200px;
    top: 26%;
    left: 4%;
    border-radius: 50%;
    background: radial-gradient(ellipse at 33% 28%,
        rgba(255, 225, 240, 1.00)  0%,
        rgba(255, 190, 225, 0.97) 14%,
        rgba(255, 155, 210, 0.90) 28%,
        rgba(248, 120, 195, 0.78) 42%,
        rgba(235,  88, 180, 0.58) 57%,
        rgba(218,  62, 165, 0.32) 72%,
        rgba(198,  42, 150, 0.10) 85%,
        rgba(178,  28, 138, 0.00) 95%
    );
}

.blob-pink-m {
    width: 280px;
    height: 280px;
    top: 16%;
    left: 16%;
    border-radius: 50%;
    background: radial-gradient(ellipse at 32% 27%,
        rgba(255, 235, 248, 1.00)  0%,
        rgba(255, 205, 238, 0.97) 12%,
        rgba(255, 168, 222, 0.90) 26%,
        rgba(252, 132, 208, 0.78) 40%,
        rgba(240,  98, 192, 0.58) 56%,
        rgba(222,  68, 178, 0.32) 72%,
        rgba(202,  45, 162, 0.10) 85%,
        rgba(180,  28, 148, 0.00) 95%
    );
}

.blob-blue-r {
    width: 240px;
    height: 240px;
    top: 22%;
    right: 5%;
    border-radius: 50%;
    background: radial-gradient(ellipse at 34% 28%,
        rgba(235, 242, 255, 1.00)  0%,
        rgba(208, 225, 255, 0.97) 14%,
        rgba(178, 205, 255, 0.90) 28%,
        rgba(148, 182, 252, 0.78) 42%,
        rgba(118, 158, 242, 0.55) 57%,
        rgba( 90, 135, 228, 0.28) 72%,
        rgba( 65, 112, 215, 0.08) 85%,
        rgba( 45,  92, 200, 0.00) 95%
    );
}

.blob-blue-b {
    width: 310px;
    height: 310px;
    bottom: 3%;
    left: 44%;
    border-radius: 50%;
    background: radial-gradient(ellipse at 34% 28%,
        rgba(228, 238, 255, 1.00)  0%,
        rgba(200, 220, 255, 0.97) 14%,
        rgba(170, 200, 255, 0.90) 28%,
        rgba(140, 178, 252, 0.78) 42%,
        rgba(110, 154, 240, 0.55) 57%,
        rgba( 82, 130, 226, 0.28) 72%,
        rgba( 58, 108, 212, 0.08) 85%,
        rgba( 38,  88, 198, 0.00) 95%
    );
}

.blob-sparkle {
    position: fixed;
    width: 130px;
    height: 130px;
    top: calc(50% + 62px);
    left: calc(50% - 12px);
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(ellipse 32% 7% at 50% 50%,
            rgba(255, 255, 255, 1.00) 0%,
            rgba(255, 255, 255, 0.00) 100%
        ),
        radial-gradient(ellipse 7% 32% at 50% 50%,
            rgba(255, 255, 255, 1.00) 0%,
            rgba(255, 255, 255, 0.00) 100%
        );
}

.stApp > * {
    position: relative;
    z-index: 1;
}

/* ── Page title ── */
h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    color: #2a1060 !important;
    text-align: center;
    text-transform: uppercase;
    text-shadow: none !important;
    margin-bottom: 8px !important;
}

/* ── Thin accent line under the page title ── */
.title-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #a78bfa, #f472b6);
    border-radius: 2px;
    margin: 0 auto 28px auto;
}

/* ── Section headers ── */
.section-header {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(58, 24, 128, 0.60);
    margin-bottom: 14px;
    margin-top: 4px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(140, 100, 200, 0.18);
}

/* ── Sidebar title ── */
.sidebar-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: rgba(58, 24, 128, 0.55);
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(140, 100, 200, 0.18);
    margin-bottom: 6px;
}

/* ── Glass card ── */
.glass-card,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed) {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(40px) saturate(160%) brightness(1.06);
    -webkit-backdrop-filter: blur(40px) saturate(160%) brightness(1.06);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.35);
    box-shadow:
        inset 0 1.5px 0 rgba(255, 255, 255, 0.60),
        inset 1px 0 0   rgba(255, 255, 255, 0.30),
        0 8px 32px rgba(140, 100, 200, 0.08),
        0 2px 8px  rgba(0,   0,   0,   0.04);
    padding: 22px;
    width: 100%;
    transition: transform 0.28s ease, box-shadow 0.28s ease;
    overflow: hidden;
}

.glass-card:hover,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed):hover {
    transform: translateY(-8px);
    box-shadow:
        inset 0 1.5px 0 rgba(255, 255, 255, 0.75),
        inset 1px 0 0   rgba(255, 255, 255, 0.40),
        0 0 55px rgba(200, 150, 255, 0.22),
        0 18px 55px rgba(0, 0, 0, 0.08);
}

.glass-card {
    display: flex;
    flex-direction: column;
}

/* ── Kill Altair white canvas ── */
.vega-embed,
.vega-embed canvas,
.vega-embed svg {
    background: transparent !important;
}

/* ── Glass HTML table ── */
.glass-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.glass-table thead tr {
    border-bottom: 1px solid rgba(150, 120, 200, 0.22);
}

.glass-table th {
    padding: 12px 16px;
    text-align: left;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: rgba(60, 30, 120, 0.50);
}

.glass-table td {
    padding: 11px 16px;
    color: #2a1060;
    border-bottom: 1px solid rgba(180, 160, 220, 0.13);
}

.glass-table tbody tr:hover td {
    background: rgba(255, 255, 255, 0.20);
}

.glass-table tbody tr:last-child td {
    border-bottom: none;
}

/* ── Metric tiles ── */
.metric-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(80, 40, 160, 0.55);
    margin-bottom: 8px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #4a0fa8;
    letter-spacing: -0.01em;
}

/* ── Inputs ── */
label,
.stSelectbox label,
.stTextInput label,
.stNumberInput label {
    color: #3a1880 !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}

</style>

<div class="blob blob-bl"></div>
<div class="blob blob-br"></div>
<div class="blob blob-center"></div>
<div class="blob blob-pink-l"></div>
<div class="blob blob-pink-m"></div>
<div class="blob blob-blue-r"></div>
<div class="blob blob-blue-b"></div>
<div class="blob-sparkle"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown(
    "<div class='sidebar-title'>Navigation</div>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    [
        "System Overview",
        "Customer Search",
        "Customer Details",
        "Risk Ranking",
        "Device / IP Risk Panel",
        "Fraud Alerts Feed"
    ]
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
    <h1>Fraud Detection Dashboard</h1>
    <div class='title-divider'></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SYSTEM OVERVIEW
# ---------------------------------------------------------
if page == "System Overview":

    st.markdown("<div class='section-header'>Fraud Metrics</div>", unsafe_allow_html=True)

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

    # Donut Chart
    pie_data = transactions_fe.copy()
    pie_data["bucket"] = pd.cut(
        pie_data["anomaly_score"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    )

    pie_chart = (
        alt.Chart(pie_data)
        .mark_arc(innerRadius=70, outerRadius=120)
        .encode(
            theta="count():Q",
            color=alt.Color("bucket:N", scale=alt.Scale(scheme="magma")),
            tooltip=["bucket:N", "count():Q"]
        )
        .properties(height=400, background="transparent")
        .configure_view(strokeWidth=0)
    )

    with st.container():
        st.markdown("<div class='section-header'>Anomaly Score Distribution</div>", unsafe_allow_html=True)
        st.altair_chart(pie_chart, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Highest-Risk Customers</div>", unsafe_allow_html=True)

    risk_df = (
        transactions_fe.groupby("customer_id")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    risk_display = risk_df.head(20).copy()
    risk_display["anomaly_score"] = risk_display["anomaly_score"].round(4)
    risk_display["risk_level"] = risk_display["anomaly_score"].apply(risk_badge)
    glass_table(risk_display)

# ---------------------------------------------------------
# CUSTOMER SEARCH
# ---------------------------------------------------------
elif page == "Customer Search":
    st.markdown("<div class='section-header'>Customer Search</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    customer_id = st.text_input("Customer ID")
    st.markdown("</div>", unsafe_allow_html=True)

    if customer_id:
        try:
            customer_id = int(customer_id)
            cust_df = transactions_fe[transactions_fe["customer_id"] == customer_id]

            if cust_df.empty:
                st.warning("No records found for this customer ID.")
            else:
                st.success(f"{len(cust_df)} transactions found.")
                glass_table(cust_df.head(20).copy())

        except:
            st.error("Please enter a valid numeric ID.")

# ---------------------------------------------------------
# CUSTOMER DETAILS
# ---------------------------------------------------------
elif page == "Customer Details":
    st.markdown("<div class='section-header'>Customer Details</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    customer_id = st.number_input("Customer ID", min_value=1, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

    cust_df = transactions_fe[transactions_fe["customer_id"] == customer_id]

    if not cust_df.empty:
        col1, col2, col3 = st.columns(3)

        avg_score = round(cust_df["anomaly_score"].mean(), 4)

        details = [
            ("Total Transactions", len(cust_df)),
            ("Avg Anomaly Score", f"{avg_score} {risk_badge(avg_score)}"),
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

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Recent Transactions</div>", unsafe_allow_html=True)
        glass_table(cust_df.tail(20).copy())

    else:
        st.info("Enter a valid customer ID to view details.")

# ---------------------------------------------------------
# RISK RANKING
# ---------------------------------------------------------
elif page == "Risk Ranking":
    st.markdown("<div class='section-header'>Highest-Risk Customers</div>", unsafe_allow_html=True)

    risk_df = (
        transactions_fe.groupby("customer_id")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    risk_display = risk_df.head(20).copy()
    risk_display["anomaly_score"] = risk_display["anomaly_score"].round(4)
    risk_display["risk_level"] = risk_display["anomaly_score"].apply(risk_badge)
    glass_table(risk_display)

# ---------------------------------------------------------
# DEVICE / IP RISK PANEL
# ---------------------------------------------------------
elif page == "Device / IP Risk Panel":
    st.markdown("<div class='section-header'>Device & IP Risk Panel</div>", unsafe_allow_html=True)

    cust_numeric = pd.to_numeric(transactions_fe["customer_id"], errors="coerce").fillna(0).astype(int)

    if "device_id" not in transactions_fe.columns:
        transactions_fe["device_id"] = (cust_numeric % 50).astype(str)

    if "ip_address" not in transactions_fe.columns:
        transactions_fe["ip_address"] = "192.168.1." + (cust_numeric % 255).astype(str)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Highest-Risk Devices</div>", unsafe_allow_html=True)

    device_risk = (
        transactions_fe.groupby("device_id")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    device_display = device_risk.head(20).copy()
    device_display["anomaly_score"] = device_display["anomaly_score"].round(4)
    device_display["risk_level"] = device_display["anomaly_score"].apply(risk_badge)
    glass_table(device_display)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Highest-Risk IP Addresses</div>", unsafe_allow_html=True)

    ip_risk = (
        transactions_fe.groupby("ip_address")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    ip_display = ip_risk.head(20).copy()
    ip_display["anomaly_score"] = ip_display["anomaly_score"].round(4)
    ip_display["risk_level"] = ip_display["anomaly_score"].apply(risk_badge)
    glass_table(ip_display)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Shared Devices — Possible Fraud Rings</div>", unsafe_allow_html=True)

    device_sharing = (
        transactions_fe.groupby("device_id")["customer_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"customer_id": "unique_customers"})
    )
    glass_table(device_sharing)

# ---------------------------------------------------------
# FRAUD ALERTS FEED
# ---------------------------------------------------------
elif page == "Fraud Alerts Feed":
    st.markdown("<div class='section-header'>Fraud Alerts Feed</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        top_n = st.slider("Number of alerts to show", min_value=10, max_value=100, value=25, step=5)
    with col_right:
        threshold = st.slider("Minimum anomaly score", min_value=0.0, max_value=1.0, value=0.7, step=0.01)

    alerts = (
        transactions_fe[transactions_fe["anomaly_score"] >= threshold]
        .copy()
        .sort_values("anomaly_score", ascending=False)
        .head(top_n)
    )

    if alerts.empty:
        st.info("No transactions found above the selected anomaly score threshold.")
    else:
        alerts_display = alerts.copy()
        alerts_display["anomaly_score"] = alerts_display["anomaly_score"].round(4)
        alerts_display["risk_level"] = alerts_display["anomaly_score"].apply(risk_badge)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>High-Risk Transactions</div>", unsafe_allow_html=True)
        glass_table(alerts_display)