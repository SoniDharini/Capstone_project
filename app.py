# app.py
import streamlit as st
from autoda import AutoDA

st.title("AutoDA Prototype - Intelligent Data Analysis")

if "engine" not in st.session_state:
    st.session_state.engine = AutoDA(workdir="assets")

engine = st.session_state.engine

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    engine.load(uploaded_file.read(), uploaded_file.name)
    profile = engine.profile()
    st.json(profile)
    target_column = st.selectbox("Select target column", profile["types"].keys())
    if st.button("Set Target"):
        engine.set_target(target_column)
        st.write(f"Task detected: {engine.task}")

    if st.button("Train Model"):
        engine.suggest_features()
        artifacts = engine.save_artifacts()
        st.write("Model Trained!")
        st.download_button("Download Report", data=open(artifacts["report"], "rb").read(), file_name="AutoDA_report.md", mime="text/markdown")
        st.download_button("Download Model", data=open(artifacts["model"], "rb").read(), file_name="AutoDA_model.pkl", mime="application/octet-stream")
