import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.title("Intelligent Fund Flow Tracking System")

# Load dataset
transactions = pd.read_csv("sample_transactions.csv")

st.subheader("Transaction Dataset Preview")
st.write(transactions.head())

# Transaction amount chart
st.subheader("Transaction Amount Distribution")

fig, ax = plt.subplots()
ax.hist(transactions["Amount Paid"], bins=30)
st.pyplot(fig)

# ML anomaly detection
model = IsolationForest(contamination=0.01)
transactions["anomaly"] = model.fit_predict(transactions[["Amount Paid"]])

# suspicious transactions
st.subheader("Suspicious Transactions")
suspicious = transactions[transactions["anomaly"] == -1]

st.write(suspicious.head(10))

# summary
st.subheader("System Summary")
st.write("Total Transactions:", len(transactions))
st.write("Suspicious Transactions:", len(suspicious))

# top accounts
st.subheader("Top Accounts by Transaction Amount")

top_accounts = transactions.groupby("Account")["Amount Paid"].sum().sort_values(ascending=False).head(10)

st.bar_chart(top_accounts)