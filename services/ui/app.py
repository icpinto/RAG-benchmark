from __future__ import annotations
import os
import requests
import pandas as pd
import streamlit as st

API=os.getenv("API_BASE_URL", "http://api:8000")


def apply_app_theme():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
        div[data-testid="stExpander"] details {
            border-radius: 0.75rem;
        }
        .rag-hero {
            background: linear-gradient(135deg, #EEF5FF 0%, #F8FBFF 48%, #F3EFFF 100%);
            border: 1px solid #DCE8FF;
            border-radius: 1.25rem;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.25rem;
        }
        .rag-hero h1 {
            margin: 0 0 .35rem 0;
            color: #12263F;
        }
        .rag-hero p {
            margin: 0;
            color: #486581;
            font-size: 1.05rem;
        }
        .section-card {
            border: 1px solid #E6ECF5;
            border-radius: 1rem;
            padding: 1rem 1.1rem;
            background: #FFFFFF;
            margin-bottom: 1rem;
        }
        .section-card h3 {
            margin-top: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_intro(title: str, description: str):
    st.markdown(
        f'<div class="section-card"><h3>{title}</h3><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


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
    st.header("1. Knowledge Base & Evaluation Questions")
    section_intro("Prepare your benchmark inputs", "Upload source documents, inspect raw files, and maintain the ground-truth questions used to score each RAG run.")
    files_tab,questions_tab=st.tabs(["📄 Source documents", "✅ Ground-truth questions"])
    with files_tab:
        try:
            listing=requests.get(f"{API}/data/files", timeout=30).json()
            files=listing.get("files", [])
            st.caption(f"Raw data directory: {listing.get('data_root', 'unknown')}")
            if files:
                st.dataframe(pd.DataFrame(files), use_container_width=True)
                selected_file=st.selectbox("Choose a source file to preview", [f["path"] for f in files])
                if selected_file:
                    try:
                        content=requests.get(f"{API}/data/files/content", params={"path":selected_file}, timeout=30).json()
                        st.text_area("Source file preview", value=content.get("content", ""), height=360, disabled=True)
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
                selected_index=st.selectbox("Choose a question to review", range(len(questions)), format_func=lambda i: f"{i + 1}. {questions[i].get('question', '')[:90]}")
                with st.form(f"edit_question_{selected_index}"):
                    edited=_question_form_payload(f"edit_{selected_index}", questions[selected_index])
                    edit_col,delete_col=st.columns(2)
                    save_clicked=edit_col.form_submit_button("Save changes")
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
            with st.expander("➕ Add a ground-truth question", expanded=not questions):
                with st.form("add_question"):
                    new_question=_question_form_payload("new")
                    add_clicked=st.form_submit_button("Add ground-truth question")
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
    st.subheader("Trace Explorer")
    detail_index=st.selectbox(
        "Choose a result to inspect",
        range(len(results)),
        format_func=lambda index: _result_label(results[index], index),
    )
    selected=results[detail_index]
    widget_key=f"{selected.get('case_id')}_{detail_index}"
    evaluation_step=_evaluation_step(selected)
    if not evaluation_step:
        st.info("This result does not include saved context or query-build details. Re-run the benchmark to capture evaluation-step traces.")
        return

    st.caption("Debug the context, retrieved chunks, and prompt/query text that produced the selected answer.")
    context_tab,query_tab,chunks_tab,config_tab=st.tabs(["🧩 Final context", "✍️ Prompt/query", "🔎 Retrieved chunks", "⚙️ Case configuration"])
    with context_tab:
        st.text_area("Final assembled context sent to the generator", value=evaluation_step.get("context", ""), height=360, disabled=True, key=f"context_{widget_key}")
    with query_tab:
        rewritten=evaluation_step.get("rewritten_queries", [])
        if rewritten:
            st.write("Rewritten search queries")
            for query in rewritten:
                st.code(query, language="text")
        st.write("Prompt or query construction")
        st.text_area("Prompt/query build details", value=evaluation_step.get("query_build", ""), height=420, disabled=True, key=f"query_build_{widget_key}")
    with chunks_tab:
        chunks=evaluation_step.get("retrieved_context", [])
        if chunks:
            chunk_rows=[{key:chunk.get(key) for key in ["rank","score","retriever","doc_id","chunk_id"]} for chunk in chunks]
            st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True)
            for chunk in chunks:
                label=f"Rank {chunk.get('rank')} · {chunk.get('doc_id')}::{chunk.get('chunk_id')}"
                with st.expander(label):
                    st.write({"score":chunk.get("score"),"retriever":chunk.get("retriever"),"metadata":chunk.get("metadata", {})})
                    st.text_area("Retrieved chunk text", value=chunk.get("text", ""), height=220, disabled=True, key=f"chunk_text_{selected.get('case_id')}_{chunk.get('chunk_id')}_{chunk.get('rank')}")
        else:
            st.info("No retrieved chunks were saved for this evaluation step.")
    with config_tab:
        st.json(selected.get("case_config", {}))

st.set_page_config(page_title="RAG Benchmark Platform", page_icon="🧭", layout="wide")
apply_app_theme()
st.markdown(
    '<div class="rag-hero"><h1>RAG Benchmark Navigator</h1><p>A clearer workflow for preparing data, configuring tests, and exploring benchmark results.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Quick Uploads")
    data_files=st.file_uploader("Add source documents", accept_multiple_files=True, type=["txt","md","markdown","pdf","html"])
    if st.button("Upload source documents") and data_files:
        files=[("files", (f.name, f.getvalue())) for f in data_files]
        r=requests.post(f"{API}/data/upload", files=files, timeout=120)
        st.write(r.json())
    gt=st.file_uploader("Replace ground-truth qa.json", type=["json"])
    if st.button("Upload ground-truth JSON") and gt:
        r=requests.post(f"{API}/data/ground-truth", files={"file":("qa.json", gt.getvalue())}, timeout=120)
        st.write(r.json())

def show_test_config_setup():
    st.header("2. Benchmark Configuration")
    section_intro("Configure and launch a run", "Load an existing YAML config, edit parameters, save changes, then start a benchmark using the selected config path.")

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

    setup_col,editor_col=st.columns([1,2])
    with setup_col:
        st.subheader("Saved configurations")
        config_files=config_listing.get("files",[])
        if config_files:
            selected_config=st.selectbox("Choose a saved YAML config", config_files, index=0)
            if st.button("Load selected YAML"):
                try:
                    load_config(selected_config, sync_path_input=True)
                    st.success(f"Loaded {st.session_state.config_path}")
                except Exception as e:
                    st.error(f"Could not load config: {e}")
        else:
            st.info("No saved configs were found.")

        config_path=st.text_input("Config path in container", key="config_path_input")
        st.session_state.config_path=config_path
        if st.button("Load from path"):
            try:
                load_config(st.session_state.config_path)
                st.success(f"Loaded {st.session_state.config_path}")
            except Exception as e:
                st.error(f"Could not load config: {e}")

        st.divider()
        st.subheader("Launch benchmark")
        if st.button("Start benchmark run"):
            if st.session_state.config_dirty:
                st.warning("Unsaved YAML edits were not used. Click Save YAML first to benchmark the edited config.")
            r=requests.post(f"{API}/benchmarks/run", json={"config_path":st.session_state.config_path}, timeout=30)
            st.success(r.json())

    with editor_col:
        st.subheader("YAML editor")
        st.caption("Save YAML edits to the shared configs volume before starting a benchmark with that path.")
        edited_content=st.text_area(
            "Editable YAML configuration",
            value=st.session_state.config_editor_content,
            height=640,
            key="config_editor_text",
        )
        st.session_state.config_dirty=edited_content != st.session_state.config_editor_content
        if st.button("Save YAML changes"):
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


def show_results_layer():
    st.header("3. Benchmark Results")
    section_intro("Analyze run performance", "Review completed and queued runs, compare metrics, inspect traces, and open the raw JSON when deeper debugging is needed.")
    try:
        runs=requests.get(f"{API}/benchmarks", timeout=30).json()
        if runs:
            st.dataframe(pd.DataFrame(runs), use_container_width=True)
            run_id=st.selectbox("Choose a benchmark run", [r["id"] for r in runs])
            if run_id:
                summary_tab,results_tab,raw_tab=st.tabs(["📊 Run summary", "🏁 Scored results", "🧾 Raw JSON"])
                with summary_tab:
                    st.json(requests.get(f"{API}/benchmarks/{run_id}", timeout=30).json())
                with results_tab:
                    results=requests.get(f"{API}/benchmarks/{run_id}/results", timeout=60).json()
                    if results:
                        rows=[]
                        for item in results:
                            row={"case_id":item["case_id"],"question":item["question"],"answer":item["answer"]}
                            row.update(item.get("metrics",{}))
                            rows.append(row)
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                        show_evaluation_step_details(results)
                    else:
                        st.info("No results for this run yet. If the run stays queued, check that the Celery worker is running and consuming the ragbench queue.")
                with raw_tab:
                    results=requests.get(f"{API}/benchmarks/{run_id}/results", timeout=60).json()
                    if results:
                        st.json(results)
                    else:
                        st.info("No raw results are available for this run yet.")
        else:
            st.info("No benchmark runs yet.")
    except Exception as e:
        st.error(f"API unavailable: {e}")


navigation_options={
    "1. Knowledge Base & Questions":"Manage source documents and scoring questions",
    "2. Configure & Run Benchmark":"Edit YAML and start a benchmark run",
    "3. Results & Trace Explorer":"Review metrics, traces, and raw JSON",
}
with st.sidebar:
    st.divider()
    st.header("Navigation")
    selected_page=st.radio(
        "Go to",
        list(navigation_options),
        format_func=lambda option: f"{option} — {navigation_options[option]}",
        label_visibility="collapsed",
    )

if selected_page == "1. Knowledge Base & Questions":
    show_data_library()
elif selected_page == "2. Configure & Run Benchmark":
    show_test_config_setup()
else:
    show_results_layer()
