import streamlit as st
import pandas as pd
import numpy as np
import json
import os, sys
from sklearn.preprocessing import StandardScaler

# ---- VERY IMPORTANT: make sure Python can see the /src folder ----
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from model import AutoencoderModel, IsolationForestModel
    from response import automated_response
except Exception as e:
    st.error(
        "Couldn't import project files. "
        "Check that your repo has 'src/model.py' and 'src/response.py' (lowercase). "
        f"Error: {e}"
    )
    st.stop()

st.title("🔐 AI Threat Detection in Cloud-Native 5G")

st.sidebar.header("⚙️ Choose Model")
model_choice = st.sidebar.selectbox("Model", ["Autoencoder", "Isolation Forest"])

uploaded_file = st.file_uploader("📂 Upload Network Flow CSV", type=["csv"])

if uploaded_file:
    # 1) read data
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Sample", df.head())

    # 2) keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.error("Your CSV must have numeric columns (e.g., bytes, packets, duration).")
        st.stop()

    X_raw = df[numeric_cols].values

    # 3) scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # 4) choose + train model (quick demo training)
    if model_choice == "Autoencoder":
        model = AutoencoderModel(input_dim=X.shape[1])
        model.train(X)   # small epochs inside the class for demo
        scores = model.score(X)
    else:
        model = IsolationForestModel()
        model.train(X)
        scores = model.score(X)

    # 5) thresholds & labels
    threshold = np.percentile(scores, 95)
    anomalies = (scores >= threshold).astype(int)
    df["anomaly"] = anomalies

    # 6) show results
    st.subheader("🚨 Detection Summary")
    st.write(df["anomaly"].value_counts())

    # 7) save alerts
    alerts = [{"type": "ddos", "src_ip": str(df.index[i])} for i in df.index[df["anomaly"] == 1]]
    with open("alerts.json", "w") as f:
        json.dump(alerts, f)

    st.download_button("⬇️ Download Alerts JSON", json.dumps(alerts), "alerts.json")

    # 8) mock response
    if st.button("⚡ Trigger Automated Response"):
        automated_response("alerts.json")
        st.success("Automated Response Executed ✅")
