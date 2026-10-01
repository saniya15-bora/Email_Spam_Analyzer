"""Streamlit UI:  streamlit run app.py"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from predict import classify, load_model

st.set_page_config(page_title="Spam & Phishing Analyzer", page_icon="🛡️")
st.title("🛡️ Email Spam & Phishing Analyzer")

if not os.path.exists("model.joblib"):
    st.error("model.joblib not found. Run `python train.py --data data/emails.csv` first.")
    st.stop()

bundle = load_model()
threshold = st.sidebar.slider("Flag threshold", 0.3, 0.95, 0.5, 0.05,
                              help="Higher = fewer false positives, more missed threats")

text = st.text_area("Paste the email (subject + body):", height=220)

if st.button("Analyze", type="primary") and text.strip():
    r = classify(text, bundle, threshold)
    safe = r["label"] in bundle["safe_labels"]
    (st.success if safe else st.error)(f"Verdict: **{r['label'].upper()}** ({r['confidence']:.1%} confidence)")

    probs = pd.DataFrame({"class": list(r["probabilities"]), "probability": list(r["probabilities"].values())})
    st.plotly_chart(px.bar(probs, x="class", y="probability", range_y=[0, 1]), use_container_width=True)

    st.subheader("Suspicious indicators")
    ind = r["indicators"]
    c1, c2, c3 = st.columns(3)
    c1.metric("URLs", ind["num_urls"])
    c2.metric("Urgency words", ind["urgency_words"])
    c3.metric("CAPS ratio", f"{ind['caps_ratio']:.0%}")
    if ind["has_ip_url"]:
        st.warning("Link points to a raw IP address.")
    if ind["has_shortener"]:
        st.warning("Uses a URL shortener that hides the destination.")
