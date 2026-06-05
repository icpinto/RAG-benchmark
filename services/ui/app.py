from __future__ import annotations
import os
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

if "config_path" not in st.session_state:
    st.session_state.config_path="/app/configs/default.yaml"
if "config_path_input" not in st.session_state:
    st.session_state.config_path_input=st.session_state.config_path
if "config_editor_content" not in st.session_state:
    st.session_state.config_editor_content=""
if "config_dirty" not in st.session_state:
    st.session_state.config_dirty=False
if "config_editor_text" not in st.session_state:
    st.session_state.config_editor_text=st.session_state.config_editor_content

def load_config(path: str, *, sync_path_input: bool = False):
    r=requests.get(f"{API}/configs/content", params={"config_path":path}, timeout=30)
    r.raise_for_status()
    payload=r.json()
    st.session_state.config_path=payload["config_path"]
    if sync_path_input:
        st.session_state.config_path_input=payload["config_path"]
    st.session_state.config_editor_content=payload["content"]
    st.session_state.config_editor_text=payload["content"]
    st.session_state.config_dirty=False

try:
    config_listing=requests.get(f"{API}/configs", timeout=30).json()
except Exception:
    config_listing={"config_root":"/app/configs","files":[]}

config_files=config_listing.get("files",[])
if config_files:
    selected_config=st.selectbox("Saved configs", config_files, index=0)
    if st.button("Load selected config"):
        try:
            load_config(selected_config, sync_path_input=True)
            st.success(f"Loaded {st.session_state.config_path}")
        except Exception as e:
            st.error(f"Could not load config: {e}")

config_path=st.text_input("Config path inside container", key="config_path_input")
st.session_state.config_path=config_path

with st.expander("Edit config YAML", expanded=True):
    st.caption("Load a YAML file, edit it, save it to the shared configs volume, then start a benchmark with that path.")
    col_load,col_save=st.columns([1,1])
    with col_load:
        if st.button("Load config from path"):
            try:
                load_config(st.session_state.config_path)
                st.success(f"Loaded {st.session_state.config_path}")
            except Exception as e:
                st.error(f"Could not load config: {e}")
    edited_content=st.text_area(
        "YAML content",
        value=st.session_state.config_editor_content,
        height=520,
        key="config_editor_text",
    )
    st.session_state.config_dirty=edited_content != st.session_state.config_editor_content
    with col_save:
        if st.button("Save YAML"):
            try:
                r=requests.put(
                    f"{API}/configs/content",
                    json={"config_path":st.session_state.config_path,"content":edited_content},
                    timeout=30,
                )
                r.raise_for_status()
                payload=r.json()
                st.session_state.config_path=payload["config_path"]
                st.session_state.config_editor_content=edited_content
                st.session_state.config_dirty=False
                st.success(f"Saved {payload['config_path']}")
            except Exception as e:
                detail=getattr(getattr(e, "response", None), "text", None)
                st.error(f"Could not save config: {detail or e}")
    if st.session_state.config_dirty:
        st.warning("You have unsaved YAML edits. Save before starting a benchmark if you want those edits to be used.")

if st.button("Start benchmark"):
    if st.session_state.config_dirty:
        st.warning("Unsaved YAML edits were not used. Click Save YAML first to benchmark the edited config.")
    r=requests.post(f"{API}/benchmarks/run", json={"config_path":st.session_state.config_path}, timeout=30)
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
                    st.info("No results for this run yet. If the run stays queued, check that the Celery worker is running and consuming the ragbench queue.")
    else:
        st.info("No benchmark runs yet.")
except Exception as e:
    st.error(f"API unavailable: {e}")
