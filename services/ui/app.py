from __future__ import annotations
import os, json, time
from pathlib import Path
import requests
import pandas as pd
import streamlit as st

API=os.getenv("API_BASE_URL", "http://api:8000")
st.set_page_config(page_title="RAG Benchmark Platform", layout="wide")
st.title("RAG Benchmark Platform")

with st.sidebar:
    st.header("Data")
    data_files=st.file_uploader("Upload local files", accept_multiple_files=True, type=["txt","md","markdown","pdf","html"])
    if st.button("Upload files") and data_files:
        files=[("files", (f.name, f.getvalue())) for f in data_files]
        r=requests.post(f"{API}/data/upload", files=files, timeout=120)
        st.write(r.json())
    gt=st.file_uploader("Upload ground-truth qa.json", type=["json"])
    if st.button("Upload ground truth") and gt:
        r=requests.post(f"{API}/data/ground-truth", files={"file":("qa.json", gt.getvalue())}, timeout=120)
        st.write(r.json())

st.header("Run Benchmark")
config_path=st.text_input("Config path inside container", "/app/configs/default.yaml")
if st.button("Start benchmark"):
    r=requests.post(f"{API}/benchmarks/run", json={"config_path":config_path}, timeout=30)
    st.success(r.json())

st.header("Runs")
try:
    runs=requests.get(f"{API}/benchmarks", timeout=30).json()
    if runs:
        st.dataframe(pd.DataFrame(runs), use_container_width=True)
        run_id=st.selectbox("Select run", [r["id"] for r in runs])
        if run_id:
            col1,col2=st.columns(2)
            with col1:
                st.subheader("Run summary")
                st.json(requests.get(f"{API}/benchmarks/{run_id}", timeout=30).json())
            with col2:
                st.subheader("Results")
                results=requests.get(f"{API}/benchmarks/{run_id}/results", timeout=60).json()
                if results:
                    rows=[]
                    for item in results:
                        row={"case_id":item["case_id"],"question":item["question"],"answer":item["answer"]}
                        row.update(item.get("metrics",{}))
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    with st.expander("Raw results"):
                        st.json(results)
    else:
        st.info("No benchmark runs yet.")
except Exception as e:
    st.error(f"API unavailable: {e}")
