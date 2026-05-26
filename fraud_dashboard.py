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

/* ── Holographic pastel base ── */
.stApp {
    background: linear-gradient(
        135deg,
        #dce8ff 0%,
        #e8d6f8 25%,
        #f5d6ee 50%,
        #d6e4f8 75%,
        #ccd8ff 100%
    );
    background-attachment: fixed;
    min-height: 100vh;
    color: #2a1060;
    font-family: 'Inter', sans-serif;
    position: relative;
    overflow: hidden;
}

/* ── Blob base styles ── */
.blob {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
}

/* Large glowing white center orb */
.blob-center {
    width: 520px;
    height: 520px;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: radial-gradient(circle at 38% 35%,
        rgba(255, 255, 255, 1.00) 0%,
        rgba(255, 220, 240, 0.80) 30%,
        rgba(220, 200, 255, 0.50) 60%,
        rgba(180, 180, 255, 0.00) 80%
    );
    filter: blur(18px);
}

/* Top-left lavender blob — pill shaped */
.blob-tl {
    width: 420px;
    height: 300px;
    top: -60px;
    left: -80px;
    border-radius: 60% 40% 70% 30% / 50% 60% 40% 50%;
    background: radial-gradient(circle at 40% 35%,
        rgba(210, 200, 255, 0.95) 0%,
        rgba(180, 170, 245, 0.75) 45%,
        rgba(150, 140, 230, 0.00) 75%
    );
    filter: blur(8px);
}

/* Top-right squiggle blob */
.blob-tr {
    width: 380px;
    height: 220px;
    top: -30px;
    right: -60px;
    border-radius: 70% 30% 50% 50% / 40% 60% 40% 60%;
    background: radial-gradient(circle at 35% 40%,
        rgba(200, 215, 255, 0.95) 0%,
        rgba(170, 190, 245, 0.70) 45%,
        rgba(140, 165, 230, 0.00) 75%
    );
    filter: blur(10px);
}

/* Left pink sphere */
.blob-left-pink {
    width: 170px;
    height: 170px;
    top: 28%;
    left: 3%;
    background: radial-gradient(circle at 35% 30%,
        rgba(255, 200, 230, 1.00) 0%,
        rgba(255, 140, 200, 0.90) 40%,
        rgba(220, 100, 180, 0.40) 70%,
        rgba(180,  80, 160, 0.00) 90%
    );
    filter: blur(2px);
    box-shadow: inset -6px -6px 14px rgba(180, 80, 160, 0.25);
}

/* Center-left hot-pink sphere */
.blob-mid-pink {
    width: 240px;
    height: 240px;
    top: 20%;
    left: 18%;
    background: radial-gradient(circle at 35% 28%,
        rgba(255, 210, 235, 1.00) 0%,
        rgba(255, 150, 210, 0.90) 38%,
        rgba(230, 100, 190, 0.45) 65%,
        rgba(190,  70, 170, 0.00) 85%
    );
    filter: blur(3px);
    box-shadow: inset -8px -8px 18px rgba(190, 70, 170, 0.30);
}

/* Right-side periwinkle sphere */
.blob-right-blue {
    width: 200px;
    height: 200px;
    top: 25%;
    right: 6%;
    background: radial-gradient(circle at 35% 30%,
        rgba(220, 230, 255, 1.00) 0%,
        rgba(170, 195, 255, 0.90) 40%,
        rgba(130, 160, 240, 0.40) 68%,
        rgba(100, 130, 220, 0.00) 88%
    );
    filter: blur(2px);
    box-shadow: inset -7px -7px 16px rgba(100, 130, 220, 0.28);
}

/* Bottom-left large lavender blob */
.blob-bl {
    width: 500px;
    height: 350px;
    bottom: -80px;
    left: -100px;
    border-radius: 50% 50% 40% 60% / 60% 40% 60% 40%;
    background: radial-gradient(circle at 40% 35%,
        rgba(200, 215, 255, 0.95) 0%,
        rgba(170, 190, 248, 0.75) 45%,
        rgba(140, 165, 235, 0.00) 75%
    );
    filter: blur(12px);
}

/* Bottom-center medium sphere */
.blob-bc {
    width: 280px;
    height: 280px;
    bottom: 2%;
    left: 42%;
    background: radial-gradient(circle at 35% 28%,
        rgba(215, 225, 255, 1.00) 0%,
        rgba(175, 195, 255, 0.90) 40%,
        rgba(140, 165, 245, 0.40) 68%,
        rgba(110, 135, 225, 0.00) 88%
    );
    filter: blur(3px);
    box-shadow: inset -9px -9px 20px rgba(110, 135, 225, 0.30);
}

/* Bottom-right large blob */
.blob-br {
    width: 440px;
    height: 320px;
    bottom: -60px;
    right: -80px;
    border-radius: 40% 60% 50% 50% / 50% 40% 60% 50%;
    background: radial-gradient(circle at 38% 32%,
        rgba(210, 205, 255, 0.95) 0%,
        rgba(180, 175, 248, 0.70) 45%,
        rgba(150, 145, 235, 0.00) 75%
    );
    filter: blur(12px);
}

/* Sparkle / lens flare on the center orb */
.blob-sparkle {
    position: fixed;
    width: 80px;
    height: 80px;
    top: calc(50% + 60px);
    left: calc(50% - 10px);
    background: radial-gradient(circle,
        rgba(255, 255, 255, 1.00) 0%,
        rgba(255, 230, 250, 0.60) 30%,
        rgba(255, 200, 240, 0.00) 70%
    );
    filter: blur(2px);
    pointer-events: none;
    z-index: 0;
}

/* Keep Streamlit content above blobs */
.stApp > * {
    position: relative;
    z-index: 1;
}

/* ── Shared liquid-glass card ── */
.glass-card,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed) {
    background: rgba(255, 255, 255, 0.20);
    backdrop-filter: blur(50px) saturate(180%) brightness(1.08);
    -webkit-backdrop-filter: blur(50px) saturate(180%) brightness(1.08);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.55);
    box-shadow:
        inset 0 1.5px 0 rgba(255, 255, 255, 0.80),
        inset 1px 0 0   rgba(255, 255, 255, 0.40),
        0 8px 32px rgba(140, 100, 200, 0.12),
        0 2px 8px  rgba(0,   0,   0,   0.06);
    padding: 22px;
    width: 100%;
    transition: transform 0.28s ease, box-shadow 0.28s ease;
    overflow: hidden;
}

.glass-card:hover,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed):hover {
    transform: translateY(-8px);
    box-shadow:
        inset 0 1.5px 0 rgba(255, 255, 255, 0.90),
        inset 1px 0 0   rgba(255, 255, 255, 0.50),
        0 0 50px rgba(200, 150, 255, 0.30),
        0 16px 50px rgba(0, 0, 0, 0.10);
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
    font-size: 14px;
}

.glass-table thead tr {
    border-bottom: 1px solid rgba(150, 120, 200, 0.25);
}

.glass-table th {
    padding: 12px 16px;
    text-align: left;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(60, 30, 120, 0.60);
}

.glass-table td {
    padding: 11px 16px;
    color: #2a1060;
    border-bottom: 1px solid rgba(180, 160, 220, 0.15);
}

.glass-table tbody tr:hover td {
    background: rgba(255, 255, 255, 0.25);
}

.glass-table tbody tr:last-child td {
    border-bottom: none;
}

/* ── Typography ── */
.section-header {
    font-size: 22px;
    font-weight: 700;
    color: #3a1880;
    margin-bottom: 12px;
    text-shadow: 0 1px 3px rgba(255, 255, 255, 0.60);
}

.metric-title {
    font-size: 11px;
    color: rgba(80, 40, 160, 0.70);
    margin-bottom: 6px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #5a12c0;
    text-shadow: 0 2px 10px rgba(140, 80, 220, 0.20);
}

h1 {
    color: #3a1880 !important;
    text-shadow: 0 2px 10px rgba(255, 255, 255, 0.50) !important;
}

label,
.stSelectbox label,
.stTextInput label,
.stNumberInput label {
    color: #3a1880 !important;
}

</style>

<!-- Blob elements injected as HTML since CSS only gives us 2 pseudo-elements -->
<div class="blob blob-center"></div>
<div class="blob blob-tl"></div>
<div class="blob blob-tr"></div>
<div class="blob blob-left-pink"></div>
<div class="blob blob-mid-pink"></div>
<div class="blob blob-right-blue"></div>
<div class="blob blob-bl"></div>
<div class="blob blob-bc"></div>
<div class="blob blob-br"></div>
<div class="blob-sparkle"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown(
    "<h2 style='color:#3a1880;'>📊 Dashboard</h2>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    ["System Overview", "Customer Search", "Customer Details", "Risk Ranking", "Device/IP Risk Panel"]
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🔍 Fraud Detection Dashboard</h1>",
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
        .properties(
            height=400,
            background="transparent"
        )
        .configure_view(strokeWidth=0)
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

    # TABLE
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔥 Highest-Risk Customers</div>", unsafe_allow_html=True)

    risk_df = (
        transactions_fe.groupby("customer_id")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    risk_display = risk_df.head(20).copy()
    risk_display["anomaly_score"] = risk_display["anomaly_score"].round(4)
    glass_table(risk_display)

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

            if cust_df.empty:
                st.warning("Customer not found.")
            else:
                st.success(f"Found {len(cust_df)} transactions.")
                glass_table(cust_df.head(20).copy())

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

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🧾 Recent Transactions</div>", unsafe_allow_html=True)
        glass_table(cust_df.tail(20).copy())

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
    risk_display = risk_df.head(20).copy()
    risk_display["anomaly_score"] = risk_display["anomaly_score"].round(4)
    glass_table(risk_display)

# ---------------------------------------------------------
# DEVICE / IP RISK PANEL
# ---------------------------------------------------------
elif page == "Device/IP Risk Panel":
    st.markdown("<div class='section-header'>🖥️ Device & IP Risk Panel</div>", unsafe_allow_html=True)

    # --- Simulated device/IP fields ---
    # If your dataset already has device_id or ip_address, replace these lines.
    if "device_id" not in transactions_fe.columns:
        transactions_fe["device_id"] = transactions_fe["customer_id"] % 50  # fake grouping
    if "ip_address" not in transactions_fe.columns:
        transactions_fe["ip_address"] = "192.168.1." + (transactions_fe["customer_id"] % 255).astype(str)

    # --- Device Risk ---
    device_risk = (
        transactions_fe.groupby("device_id")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .head(20)
    )

    with st.container():
        st.markdown("<div class='section-header'>⚠️ Highest-Risk Devices</div>", unsafe_allow_html=True)
        st.dataframe(device_risk, use_container_width=True)

    # --- IP Risk ---
    ip_risk = (
        transactions_fe.groupby("ip_address")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .head(20)
    )

    with st.container():
        st.markdown("<div class='section-header'>🌐 Highest-Risk IP Addresses</div>", unsafe_allow_html=True)
        st.dataframe(ip_risk, use_container_width=True)

    # --- Device Sharing ---
    device_sharing = (
        transactions_fe.groupby("device_id")["customer_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"customer_id": "unique_customers"})
        .head(20)
    )

    with st.container():
        st.markdown("<div class='section-header'>🔗 Shared Devices (Possible Fraud Rings)</div>", unsafe_allow_html=True)
        st.dataframe(device_sharing, use_container_width=True)
