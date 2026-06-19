from __future__ import annotations
import os
import requests
import pandas as pd
import streamlit as st

API=os.getenv("API_BASE_URL", "http://api:8000")


def api_request(method: str, path: str, **kwargs):
    """Call the API and surface connection/HTTP failures in the UI."""
    try:
        response=requests.request(method, f"{API}{path}", **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        detail=getattr(getattr(exc, "response", None), "text", None)
        message=detail or str(exc)
        st.error(f"API request failed for {path}: {message}")
        return None


def api_json(method: str, path: str, *, default=None, **kwargs):
    response=api_request(method, path, **kwargs)
    if response is None:
        return default
    try:
        return response.json()
    except ValueError as exc:
        st.error(f"API returned invalid JSON for {path}: {exc}")
        return default


def _result_label(item: dict, index: int) -> str:
    question=item.get("question", "")
    case_id=item.get("case_id", "unknown case")
    question_preview=question[:80] + ("…" if len(question)>80 else "")
    return f"{index + 1}. {case_id} — {question_preview}"

def _evaluation_step(item: dict) -> dict:
    return item.get("case_config", {}).get("evaluation_step", {}) or {}

def show_evaluation_step_details(results: list[dict]):
    st.subheader("Evaluation step details")
    detail_index=st.selectbox(
        "Select evaluation step",
        range(len(results)),
        format_func=lambda index: _result_label(results[index], index),
    )
    selected=results[detail_index]
    widget_key=f"{selected.get('case_id')}_{detail_index}"
    evaluation_step=_evaluation_step(selected)
    if not evaluation_step:
        st.info("This result does not include saved context or query-build details. Re-run the benchmark to capture evaluation-step traces.")
        return

    st.caption("Inspect the exact context and prompt/query build used to generate this result.")
    context_tab,query_tab,chunks_tab,config_tab=st.tabs(["Context", "Query build", "Retrieved chunks", "Case config"])
    with context_tab:
        st.text_area("Assembled context", value=evaluation_step.get("context", ""), height=360, disabled=True, key=f"context_{widget_key}")
    with query_tab:
        rewritten=evaluation_step.get("rewritten_queries", [])
        if rewritten:
            st.write("Rewritten queries")
            for query in rewritten:
                st.code(query, language="text")
        st.write("Prompt / query build")
        st.text_area("Query build", value=evaluation_step.get("query_build", ""), height=420, disabled=True, key=f"query_build_{widget_key}")
    with chunks_tab:
        chunks=evaluation_step.get("retrieved_context", [])
        if chunks:
            chunk_rows=[{key:chunk.get(key) for key in ["rank","score","retriever","doc_id","chunk_id"]} for chunk in chunks]
            st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True)
            for chunk in chunks:
                label=f"Rank {chunk.get('rank')} · {chunk.get('doc_id')}::{chunk.get('chunk_id')}"
                with st.expander(label):
                    st.write({"score":chunk.get("score"),"retriever":chunk.get("retriever"),"metadata":chunk.get("metadata", {})})
                    st.text_area("Chunk text", value=chunk.get("text", ""), height=220, disabled=True, key=f"chunk_text_{selected.get('case_id')}_{chunk.get('chunk_id')}_{chunk.get('rank')}")
        else:
            st.info("No retrieved chunks were saved for this evaluation step.")
    with config_tab:
        st.json(selected.get("case_config", {}))

st.set_page_config(page_title="RAG Benchmark Platform", layout="wide")
st.title("RAG Benchmark Platform")

with st.sidebar:
    st.header("Data")
    data_files=st.file_uploader("Upload local files", accept_multiple_files=True, type=["txt","md","markdown","pdf","html"])
    if st.button("Upload files") and data_files:
        files=[("files", (f.name, f.getvalue())) for f in data_files]
        payload=api_json("POST", "/data/upload", files=files, timeout=120)
        if payload is not None:
            st.write(payload)
    gt=st.file_uploader("Upload ground-truth qa.json", type=["json"])
    if st.button("Upload ground truth") and gt:
        payload=api_json("POST", "/data/ground-truth", files={"file":("qa.json", gt.getvalue())}, timeout=120)
        if payload is not None:
            st.write(payload)

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

def load_config(path: str, *, sync_path_input: bool = False) -> bool:
    payload=api_json("GET", "/configs/content", params={"config_path":path}, timeout=30)
    if payload is None:
        return False
    st.session_state.config_path=payload["config_path"]
    if sync_path_input:
        st.session_state.config_path_input=payload["config_path"]
    st.session_state.config_editor_content=payload["content"]
    st.session_state.config_editor_text=payload["content"]
    st.session_state.config_dirty=False
    return True

config_listing=api_json("GET", "/configs", timeout=30, default={"config_root":"/app/configs","files":[]})

config_files=config_listing.get("files",[])
if config_files:
    selected_config=st.selectbox("Saved configs", config_files, index=0)
    if st.button("Load selected config"):
        try:
            if load_config(selected_config, sync_path_input=True):
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
                if load_config(st.session_state.config_path):
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
                payload=api_json(
                    "PUT",
                    "/configs/content",
                    json={"config_path":st.session_state.config_path,"content":edited_content},
                    timeout=30,
                )
                if payload is None:
                    st.stop()
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
    payload=api_json("POST", "/benchmarks/run", json={"config_path":st.session_state.config_path}, timeout=30)
    if payload is not None:
        st.success(payload)

st.header("Runs")
try:
    runs=api_json("GET", "/benchmarks", timeout=30, default=[])
    if runs:
        st.dataframe(pd.DataFrame(runs), use_container_width=True)
        run_id=st.selectbox("Select run", [r["id"] for r in runs])
        if run_id:
            col1,col2=st.columns(2)
            with col1:
                st.subheader("Run summary")
                st.json(api_json("GET", f"/benchmarks/{run_id}", timeout=30, default={}))
            with col2:
                st.subheader("Results")
                results=api_json("GET", f"/benchmarks/{run_id}/results", timeout=60, default=[])
                if results:
                    rows=[]
                    for item in results:
                        row={"case_id":item["case_id"],"question":item["question"],"answer":item["answer"]}
                        row.update(item.get("metrics",{}))
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    show_evaluation_step_details(results)
                    with st.expander("Raw results"):
                        st.json(results)
                else:
                    st.info("No results for this run yet. If the run stays queued, check that the Celery worker is running and consuming the ragbench queue.")
    else:
        st.info("No benchmark runs yet.")
except Exception as e:
    st.error(f"API unavailable: {e}")
