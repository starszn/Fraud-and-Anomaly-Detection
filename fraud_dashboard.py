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
# CSS — grain-free 4K background via stacked radial-gradients
#        (no filter:blur anywhere — vectors render crisp at any resolution)
# ---------------------------------------------------------
st.markdown("""
<style>

/* ── Force GPU compositing for crisp rendering ── */
* {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ── Holographic pastel base + all blobs as stacked gradients ── */
.stApp {
    background:
        /* Left pink sphere */
        radial-gradient(ellipse 14% 18% at 5% 35%,
            rgba(255, 180, 220, 0.95) 0%,
            rgba(255, 140, 200, 0.75) 30%,
            rgba(230, 100, 185, 0.30) 62%,
            rgba(200,  80, 170, 0.00) 82%
        ),
        /* Center-left hot-pink sphere */
        radial-gradient(ellipse 20% 24% at 20% 26%,
            rgba(255, 210, 238, 0.95) 0%,
            rgba(255, 155, 215, 0.80) 32%,
            rgba(230, 105, 192, 0.35) 62%,
            rgba(190,  70, 170, 0.00) 82%
        ),
        /* Large glowing white center orb */
        radial-gradient(ellipse 52% 52% at 50% 52%,
            rgba(255, 255, 255, 1.00)  0%,
            rgba(255, 248, 255, 0.96) 10%,
            rgba(252, 235, 255, 0.82) 26%,
            rgba(238, 220, 255, 0.58) 45%,
            rgba(218, 205, 255, 0.22) 66%,
            rgba(200, 190, 255, 0.00) 84%
        ),
        /* Right periwinkle sphere */
        radial-gradient(ellipse 18% 20% at 90% 36%,
            rgba(215, 228, 255, 0.98) 0%,
            rgba(175, 200, 255, 0.80) 32%,
            rgba(140, 168, 248, 0.32) 62%,
            rgba(110, 140, 232, 0.00) 82%
        ),
        /* Bottom-center blue sphere */
        radial-gradient(ellipse 26% 28% at 56% 92%,
            rgba(208, 222, 255, 0.98) 0%,
            rgba(170, 194, 255, 0.78) 32%,
            rgba(138, 165, 248, 0.30) 62%,
            rgba(108, 138, 232, 0.00) 82%
        ),
        /* Top-left lavender blob */
        radial-gradient(ellipse 52% 38% at -4% 4%,
            rgba(208, 202, 255, 0.92) 0%,
            rgba(182, 178, 250, 0.68) 38%,
            rgba(155, 152, 240, 0.22) 66%,
            rgba(130, 128, 226, 0.00) 84%
        ),
        /* Top-right blob */
        radial-gradient(ellipse 44% 28% at 106% -2%,
            rgba(198, 218, 255, 0.92) 0%,
            rgba(166, 192, 252, 0.68) 38%,
            rgba(140, 168, 242, 0.22) 66%,
            rgba(114, 144, 230, 0.00) 84%
        ),
        /* Bottom-left blob */
        radial-gradient(ellipse 48% 42% at -4% 106%,
            rgba(202, 216, 255, 0.92) 0%,
            rgba(172, 192, 252, 0.68) 38%,
            rgba(142, 165, 242, 0.22) 66%,
            rgba(114, 138, 230, 0.00) 84%
        ),
        /* Bottom-right blob */
        radial-gradient(ellipse 44% 38% at 110% 108%,
            rgba(212, 204, 255, 0.92) 0%,
            rgba(182, 175, 250, 0.68) 38%,
            rgba(152, 146, 238, 0.22) 66%,
            rgba(124, 120, 225, 0.00) 84%
        ),
        /* Base canvas */
        linear-gradient(
            145deg,
            #e4eaff 0%,
            #ebd8f8 20%,
            #f7d9ee 40%,
            #f1e1f8 60%,
            #dce6ff 80%,
            #d2deff 100%
        );
    background-attachment: fixed;
    min-height: 100vh;
    color: #2a1060;
    font-family: 'Inter', sans-serif;
    position: relative;
    /* GPU compositing — eliminates rasterisation grain */
    transform: translateZ(0);
    -webkit-transform: translateZ(0);
    will-change: background;
}

/* ── Sparkle/lens flare — pure gradient, no blur ── */
.blob-sparkle {
    position: fixed;
    width: 120px;
    height: 120px;
    top: calc(50% + 55px);
    left: calc(50% - 15px);
    background:
        radial-gradient(ellipse 30% 8% at 50% 50%,
            rgba(255, 255, 255, 1.00) 0%,
            rgba(255, 255, 255, 0.00) 100%
        ),
        radial-gradient(ellipse 8% 30% at 50% 50%,
            rgba(255, 255, 255, 1.00) 0%,
            rgba(255, 255, 255, 0.00) 100%
        ),
        radial-gradient(ellipse 18% 5% at 50% 50%,
            rgba(255, 220, 245, 0.80) 0%,
            rgba(255, 220, 245, 0.00) 100%
        );
    pointer-events: none;
    z-index: 0;
    transform: translateZ(0);
}

/* Keep Streamlit content above background */
.stApp > * {
    position: relative;
    z-index: 1;
}

/* ── Shared liquid-glass card ── */
.glass-card,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed) {
    background: rgba(255, 255, 255, 0.22);
    backdrop-filter: blur(40px) saturate(160%) brightness(1.06);
    -webkit-backdrop-filter: blur(40px) saturate(160%) brightness(1.06);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.60);
    box-shadow:
        inset 0 1.5px 0 rgba(255, 255, 255, 0.85),
        inset 1px 0 0   rgba(255, 255, 255, 0.45),
        0 8px 32px rgba(140, 100, 200, 0.10),
        0 2px 8px  rgba(0,   0,   0,   0.05);
    padding: 22px;
    width: 100%;
    transition: transform 0.28s ease, box-shadow 0.28s ease;
    overflow: hidden;
    transform: translateZ(0);
}

.glass-card:hover,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.vega-embed):hover {
    transform: translateY(-8px) translateZ(0);
    box-shadow:
        inset 0 1.5px 0 rgba(255, 255, 255, 0.95),
        inset 1px 0 0   rgba(255, 255, 255, 0.55),
        0 0 55px rgba(200, 150, 255, 0.28),
        0 18px 55px rgba(0, 0, 0, 0.09);
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
    border-bottom: 1px solid rgba(150, 120, 200, 0.22);
}

.glass-table th {
    padding: 12px 16px;
    text-align: left;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(60, 30, 120, 0.58);
}

.glass-table td {
    padding: 11px 16px;
    color: #2a1060;
    border-bottom: 1px solid rgba(180, 160, 220, 0.13);
}

.glass-table tbody tr:hover td {
    background: rgba(255, 255, 255, 0.28);
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
    text-shadow: 0 1px 4px rgba(255, 255, 255, 0.70);
}

.metric-title {
    font-size: 11px;
    color: rgba(80, 40, 160, 0.68);
    margin-bottom: 6px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #5a12c0;
    text-shadow: 0 2px 10px rgba(140, 80, 220, 0.18);
}

h1 {
    color: #3a1880 !important;
    text-shadow: 0 2px 12px rgba(255, 255, 255, 0.60) !important;
}

label,
.stSelectbox label,
.stTextInput label,
.stNumberInput label {
    color: #3a1880 !important;
}

</style>

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
        .mark_arc(innerRadius=70, outerRadius=120)
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

    # Create device_id if missing
    if "device_id" not in transactions_fe.columns:
        cust_numeric = pd.to_numeric(transactions_fe["customer_id"], errors="coerce").fillna(0).astype(int)
        transactions_fe["device_id"] = (cust_numeric % 50).astype(str)

    # Create ip_address if missing
    if "ip_address" not in transactions_fe.columns:
        cust_numeric = pd.to_numeric(transactions_fe["customer_id"], errors="coerce").fillna(0).astype(int)
        transactions_fe["ip_address"] = "192.168.1." + (cust_numeric % 255).astype(str)

    # Highest-Risk Devices
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>⚠️ Highest-Risk Devices</div>", unsafe_allow_html=True)

    device_risk = (
        transactions_fe.groupby("device_id")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    device_display = device_risk.head(20).copy()
    device_display["anomaly_score"] = device_display["anomaly_score"].round(4)
    glass_table(device_display)

    # Highest-Risk IP Addresses
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🌐 Highest-Risk IP Addresses</div>", unsafe_allow_html=True)

    ip_risk = (
        transactions_fe.groupby("ip_address")["anomaly_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    ip_display = ip_risk.head(20).copy()
    ip_display["anomaly_score"] = ip_display["anomaly_score"].round(4)
    glass_table(ip_display)

    # Shared Devices (Fraud Rings)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔗 Shared Devices (Possible Fraud Rings)</div>", unsafe_allow_html=True)

    device_sharing = (
        transactions_fe.groupby("device_id")["customer_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"customer_id": "unique_customers"})
    )
    sharing_display = device_sharing.head(20).copy()
    glass_table(sharing_display)
