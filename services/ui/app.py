from __future__ import annotations
import os
import requests
import pandas as pd
import streamlit as st

API=os.getenv("API_BASE_URL", "http://api:8000")


def _result_label(item: dict, index: int) -> str:
    question=item.get("question", "")
    case_id=item.get("case_id", "unknown case")
    question_preview=question[:80] + ("…" if len(question)>80 else "")
    return f"{index + 1}. {case_id} — {question_preview}"

def _evaluation_step(item: dict) -> dict:
    return item.get("case_config", {}).get("evaluation_step", {}) or {}



def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

def _question_form_payload(prefix: str, question: dict | None = None) -> dict:
    question=question or {}
    return {
        "question":st.text_area("Question", value=question.get("question", ""), height=90, key=f"{prefix}_question"),
        "expected_answer":st.text_area("Expected answer", value=question.get("expected_answer", ""), height=120, key=f"{prefix}_expected_answer"),
        "ground_truth_doc_ids":_split_csv(st.text_input("Ground-truth document IDs (comma-separated)", value=", ".join(question.get("ground_truth_doc_ids", [])), key=f"{prefix}_doc_ids")),
        "ground_truth_chunk_ids":_split_csv(st.text_input("Ground-truth chunk IDs (comma-separated)", value=", ".join(question.get("ground_truth_chunk_ids", [])), key=f"{prefix}_chunk_ids")),
        "expected_citations":_split_csv(st.text_input("Expected citations (comma-separated)", value=", ".join(question.get("expected_citations", [])), key=f"{prefix}_citations")),
    }

def show_data_library():
    st.header("Evaluation data")
    st.caption("Review the raw data files and the question set that will be used by the benchmark ground-truth file.")
    files_tab,questions_tab=st.tabs(["Data files", "Evaluation questions"])
    with files_tab:
        try:
            listing=requests.get(f"{API}/data/files", timeout=30).json()
            files=listing.get("files", [])
            st.caption(f"Raw data directory: {listing.get('data_root', 'unknown')}")
            if files:
                st.dataframe(pd.DataFrame(files), use_container_width=True)
                selected_file=st.selectbox("Preview data file", [f["path"] for f in files])
                if selected_file:
                    try:
                        content=requests.get(f"{API}/data/files/content", params={"path":selected_file}, timeout=30).json()
                        st.text_area("File content", value=content.get("content", ""), height=360, disabled=True)
                    except Exception as e:
                        st.warning(f"Could not preview file: {e}")
            else:
                st.info("No raw data files have been uploaded yet.")
        except Exception as e:
            st.error(f"Could not load data files: {e}")
    with questions_tab:
        try:
            payload=requests.get(f"{API}/data/questions", timeout=30).json()
            questions=payload.get("questions", [])
            st.caption(f"Ground-truth file: {payload.get('path', 'unknown')}")
            if questions:
                rows=[]
                for index,item in enumerate(questions):
                    rows.append({
                        "index":index,
                        "question":item.get("question", ""),
                        "expected_answer":item.get("expected_answer", ""),
                        "ground_truth_doc_ids":", ".join(item.get("ground_truth_doc_ids", [])),
                        "expected_citations":", ".join(item.get("expected_citations", [])),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                selected_index=st.selectbox("Edit or delete question", range(len(questions)), format_func=lambda i: f"{i + 1}. {questions[i].get('question', '')[:90]}")
                with st.form(f"edit_question_{selected_index}"):
                    edited=_question_form_payload(f"edit_{selected_index}", questions[selected_index])
                    edit_col,delete_col=st.columns(2)
                    save_clicked=edit_col.form_submit_button("Save question")
                    delete_clicked=delete_col.form_submit_button("Delete question")
                if save_clicked:
                    r=requests.put(f"{API}/data/questions/{selected_index}", json=edited, timeout=30)
                    r.raise_for_status()
                    st.success("Question saved")
                    st.rerun()
                if delete_clicked:
                    r=requests.delete(f"{API}/data/questions/{selected_index}", timeout=30)
                    r.raise_for_status()
                    st.success("Question deleted")
                    st.rerun()
            else:
                st.info("No evaluation questions are available yet.")
            with st.expander("Add a new evaluation question", expanded=not questions):
                with st.form("add_question"):
                    new_question=_question_form_payload("new")
                    add_clicked=st.form_submit_button("Add question")
                if add_clicked:
                    if not new_question["question"].strip():
                        st.error("Question text is required.")
                    else:
                        r=requests.post(f"{API}/data/questions", json=new_question, timeout=30)
                        r.raise_for_status()
                        st.success("Question added")
                        st.rerun()
        except Exception as e:
            st.error(f"Could not load evaluation questions: {e}")

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
        r=requests.post(f"{API}/data/upload", files=files, timeout=120)
        st.write(r.json())
    gt=st.file_uploader("Upload ground-truth qa.json", type=["json"])
    if st.button("Upload ground truth") and gt:
        r=requests.post(f"{API}/data/ground-truth", files={"file":("qa.json", gt.getvalue())}, timeout=120)
        st.write(r.json())

show_data_library()

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
                    show_evaluation_step_details(results)
                    with st.expander("Raw results"):
                        st.json(results)
                else:
                    st.info("No results for this run yet. If the run stays queued, check that the Celery worker is running and consuming the ragbench queue.")
    else:
        st.info("No benchmark runs yet.")
except Exception as e:
    st.error(f"API unavailable: {e}")
