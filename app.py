import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import IsolationForest
import os

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="BankSense AI: Intelligent Banking Analytics & Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main {background:#f7f9fc;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.3rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg,#0b1220,#172554);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {margin:0 0 .35rem 0; font-size:2rem;}
    .hero p {margin:0; color:#cbd5e1;}
    .metric-card {
        background:white;
        border:1px solid #e5eaf1;
        border-radius:14px;
        padding:1rem 1.1rem;
        box-shadow:0 4px 16px rgba(15,23,42,.04);
    }
    .metric-label {color:#64748b; font-size:.82rem;}
    .metric-value {font-size:1.65rem; font-weight:750; color:#0f172a;}
    .metric-note {font-size:.72rem; color:#16a34a;}
    .section-card {
        background:white;
        border:1px solid #e5eaf1;
        border-radius:14px;
        padding:1rem 1.1rem;
        margin-bottom:1rem;
    }
    .risk-high {
        background:#fff1f2;
        border-left:5px solid #ef4444;
        padding:.8rem 1rem;
        border-radius:9px;
    }
    .risk-ok {
        background:#ecfdf5;
        border-left:5px solid #22c55e;
        padding:.8rem 1rem;
        border-radius:9px;
    }
    [data-testid="stSidebar"] {background:#0b1220;}
    [data-testid="stSidebar"] * {color:#e2e8f0;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
@st.cache_data
def load_data():
    candidates = [
        "sample_transactions.csv",
        "data/HI-Medium_Trans.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            if path.endswith("HI-Medium_Trans.csv"):
                return pd.read_csv(path, nrows=10000)
            return pd.read_csv(path)
    raise FileNotFoundError(
        "Dataset not found. Keep sample_transactions.csv in the same folder as app.py."
    )

transactions = load_data()

# Required columns from the project dataset
required = ["Account", "Account.1", "Amount Paid"]
missing = [c for c in required if c not in transactions.columns]
if missing:
    st.error(f"Required columns missing from dataset: {missing}")
    st.stop()

transactions["Amount Paid"] = pd.to_numeric(
    transactions["Amount Paid"], errors="coerce"
).fillna(0)

# ---------------------------------------------------------
# ML anomaly detection — same approach as the project
# ---------------------------------------------------------
@st.cache_data
def detect_anomalies(df):
    model = IsolationForest(
        contamination=0.01,
        random_state=42
    )
    result = df.copy()
    result["anomaly"] = model.fit_predict(result[["Amount Paid"]])
    return result

transactions = detect_anomalies(transactions)
suspicious = transactions[transactions["anomaly"] == -1].copy()

total_transactions = len(transactions)
suspicious_count = len(suspicious)
suspicious_pct = (suspicious_count / total_transactions * 100) if total_transactions else 0
total_amount = transactions["Amount Paid"].sum()

# ---------------------------------------------------------
# Graph — fund flow representation
# ---------------------------------------------------------
@st.cache_data
def build_graph(df):
    G = nx.from_pandas_edgelist(
        df,
        source="Account",
        target="Account.1",
        edge_attr="Amount Paid",
        create_using=nx.DiGraph()
    )
    return G

G = build_graph(transactions)

# Use a manageable graph for visualisation.
top_nodes = (
    transactions.groupby("Account")["Amount Paid"]
    .sum()
    .nlargest(30)
    .index
    .tolist()
)
graph_view = G.subgraph(top_nodes).copy()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("## 🛡️ BankSense AI")
st.sidebar.caption("INTELLIGENT BANKING ANALYTICS & FRAUD DETECTION")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Fraud Detection", "Fund Flow", "Insights"],
)

st.sidebar.divider()
st.sidebar.success("● System Online")
st.sidebar.caption("ML engine active")
st.sidebar.caption(f"{total_transactions:,} transactions loaded")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Intelligent Banking Analytics & Fraud Detection</h1>
    <p>AI-powered banking transaction monitoring and fraud detection</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------
if page == "Dashboard":
    st.subheader("Fraud Intelligence Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Transactions</div>'
            f'<div class="metric-value">{total_transactions:,}</div>'
            f'<div class="metric-note">Dataset records analysed</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Transaction Volume</div>'
            f'<div class="metric-value">${total_amount:,.0f}</div>'
            f'<div class="metric-note">Total amount processed</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Suspicious Transactions</div>'
            f'<div class="metric-value">{suspicious_count:,}</div>'
            f'<div class="metric-note">Isolation Forest anomalies</div></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Suspicious Rate</div>'
            f'<div class="metric-value">{suspicious_pct:.2f}%</div>'
            f'<div class="metric-note">Requires investigation</div></div>',
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("### Transaction Amount Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(transactions["Amount Paid"], bins=40)
        ax.set_xlabel("Transaction Amount")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=.18)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("### Anomaly Distribution")
        counts = transactions["anomaly"].map({1: "Normal", -1: "Suspicious"}).value_counts()
        st.bar_chart(counts)

    st.markdown("### Recent Suspicious Transactions")
    display_cols = [c for c in [
        "Timestamp", "From Bank", "Account", "To Bank",
        "Account.1", "Amount Paid", "Receiving Currency"
    ] if c in suspicious.columns]
    st.dataframe(
        suspicious[display_cols].head(10),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------
# Fraud Detection
# ---------------------------------------------------------
elif page == "Fraud Detection":
    st.subheader("AI Fraud Detection")
    st.caption(
        "Isolation Forest identifies transactions whose amount behaviour is unusual "
        "relative to the analysed dataset."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Analysed", f"{total_transactions:,}")
    c2.metric("Suspicious", f"{suspicious_count:,}")
    c3.metric("Suspicious Rate", f"{suspicious_pct:.2f}%")

    st.markdown("### Suspicious Transactions")
    display_cols = [c for c in [
        "Timestamp", "From Bank", "Account", "To Bank",
        "Account.1", "Amount Paid", "Receiving Currency"
    ] if c in suspicious.columns]

    st.dataframe(
        suspicious[display_cols].head(50),
        use_container_width=True,
        hide_index=True
    )

    csv = suspicious.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Suspicious Transactions",
        data=csv,
        file_name="suspicious_transactions.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# Fund Flow
# ---------------------------------------------------------
elif page == "Fund Flow":
    st.subheader("Fund Flow Network")
    st.caption(
        "Each account is represented as a node and each transaction as a directed edge."
    )

    g1, g2, g3 = st.columns(3)
    g1.metric("Graph Nodes", G.number_of_nodes())
    g2.metric("Graph Transactions", G.number_of_edges())
    g3.metric("Visualised Accounts", len(graph_view.nodes))

    st.markdown("### High-Value Account Network")

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(graph_view, seed=42, k=1.0)

    suspicious_accounts = set(suspicious["Account"].astype(str))
    node_colors = [
        "red" if str(n) in suspicious_accounts else "skyblue"
        for n in graph_view.nodes()
    ]

    nx.draw_networkx_edges(
        graph_view, pos, ax=ax,
        arrows=True, alpha=.25,
        edge_color="gray",
        arrowsize=8
    )
    nx.draw_networkx_nodes(
        graph_view, pos, ax=ax,
        node_size=180,
        node_color=node_colors,
        alpha=.9
    )
    ax.set_title("Fund Flow Network — Red: Suspicious Accounts")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.info(
        "The graph view helps investigators understand how money moves "
        "between accounts and where suspicious accounts are connected."
    )

# ---------------------------------------------------------
# Insights
# ---------------------------------------------------------
else:
    st.subheader("Business & Risk Insights")

    top_accounts = (
        transactions.groupby("Account")["Amount Paid"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    top_banks = (
        transactions.groupby("From Bank")["Amount Paid"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    a, b = st.columns(2)

    with a:
        st.markdown("### Top Accounts by Transaction Amount")
        st.bar_chart(top_accounts)

    with b:
        st.markdown("### Top Banks by Transaction Volume")
        st.bar_chart(top_banks)

    st.markdown("### System Summary")
    s1, s2 = st.columns(2)
    s1.metric("Total Accounts", G.number_of_nodes())
    s2.metric("Suspicious Transactions", suspicious_count)

    suspicious_accounts = (
        suspicious["Account"]
        .value_counts()
        .head(10)
        .rename("Suspicious Transactions")
    )

    st.markdown("### Top Suspicious Accounts")
    st.dataframe(
        suspicious_accounts.to_frame(),
        use_container_width=True
    )

    if suspicious_count > 0:
        st.markdown(
            '<div class="risk-high"><b>⚠ Investigation Required</b><br>'
            'The model has identified anomalous transaction activity. '
            'These results should support, not replace, investigator review.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="risk-ok"><b>✓ No anomalies detected</b><br>'
            'No transactions were flagged by the current model configuration.</div>',
            unsafe_allow_html=True
        )

st.divider()
st.caption(
    "BankSense AI • Intelligent Banking Analytics & Fraud Detection • "
    "Machine Learning + Graph Analytics"
)
