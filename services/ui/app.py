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



RETRIEVAL_METRIC_KEYWORDS=("recall","precision","mrr","ndcg","hit","retrieval","citation")

def _short_question_label(question: str, index: int, max_length: int = 48) -> str:
    question_text=(question or "").strip().replace("\n", " ")
    if not question_text:
        return f"Q{index + 1}"
    shortened=question_text[:max_length] + ("…" if len(question_text)>max_length else "")
    return f"Q{index + 1}: {shortened}"

def _results_metrics_frame(results: list[dict]) -> pd.DataFrame:
    rows=[]
    for index,item in enumerate(results):
        row={
            "case_id":item.get("case_id", ""),
            "question_index":index + 1,
            "question":item.get("question", ""),
            "question_label":_short_question_label(item.get("question", ""), index),
        }
        row.update(item.get("metrics", {}) or {})
        rows.append(row)
    return pd.DataFrame(rows)

def _retrieval_metric_columns(frame: pd.DataFrame) -> list[str]:
    metric_columns=[]
    metadata_columns={"case_id","question_index","question","question_label","answer"}
    for column in frame.columns:
        column_name=str(column).lower()
        if column in metadata_columns:
            continue
        if not any(keyword in column_name for keyword in RETRIEVAL_METRIC_KEYWORDS):
            continue
        numeric_values=pd.to_numeric(frame[column], errors="coerce")
        if numeric_values.notna().any():
            metric_columns.append(column)
    return metric_columns

def show_retrieval_heatmaps(results: list[dict]):
    st.subheader("Retrieval heatmaps")
    st.caption("Inspect retrieval-oriented metrics by benchmark case and question.")
    if not results:
        st.info("No results are available for the selected run yet.")
        return

    frame=_results_metrics_frame(results)
    if frame.empty:
        st.info("No result rows are available to build a retrieval heatmap.")
        return

    retrieval_metrics=_retrieval_metric_columns(frame)
    if not retrieval_metrics:
        st.info("No retrieval-oriented numeric metrics were found. Expected metric names containing terms such as recall, precision, mrr, ndcg, hit, retrieval, or citation.")
        return

    selected_metric=st.selectbox("Heatmap metric", retrieval_metrics)
    heatmap_frame=frame.copy()
    heatmap_frame[selected_metric]=pd.to_numeric(heatmap_frame[selected_metric], errors="coerce")
    pivot=heatmap_frame.pivot_table(
        index="case_id",
        columns="question_label",
        values=selected_metric,
        aggfunc="mean",
    )
    if pivot.empty:
        st.info(f"No numeric values are available for {selected_metric}.")
        return

    try:
        st.dataframe(pivot.style.background_gradient(axis=None), use_container_width=True)
    except Exception:
        st.dataframe(pivot, use_container_width=True)

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


def render_quick_uploads(key_prefix: str, *, rerun_after_upload: bool = False):
    data_files=st.file_uploader(
        "Add source documents",
        accept_multiple_files=True,
        type=["txt","md","markdown","pdf","html"],
        key=f"{key_prefix}_source_documents",
    )
    if st.button("Upload source documents", key=f"{key_prefix}_upload_source_documents"):
        if not data_files:
            st.warning("Choose one or more source documents before uploading.")
        else:
            files=[("files", (f.name, f.getvalue())) for f in data_files]
            r=requests.post(f"{API}/data/upload", files=files, timeout=120)
            r.raise_for_status()
            st.success(r.json())
            if rerun_after_upload:
                st.rerun()

    gt=st.file_uploader(
        "Replace ground-truth qa.json",
        type=["json"],
        key=f"{key_prefix}_ground_truth_json",
    )
    if st.button("Upload ground-truth JSON", key=f"{key_prefix}_upload_ground_truth_json"):
        if not gt:
            st.warning("Choose a qa.json file before uploading.")
        else:
            r=requests.post(f"{API}/data/ground-truth", files={"file":("qa.json", gt.getvalue())}, timeout=120)
            r.raise_for_status()
            st.success(r.json())
            if rerun_after_upload:
                st.rerun()

def show_data_library():
    st.header("1. Knowledge Base & Evaluation Questions")
    section_intro("Prepare your benchmark inputs", "Upload source documents, inspect raw files, and maintain the ground-truth questions used to score each RAG run.")
    with st.expander("⚡ Quick uploads", expanded=True):
        st.caption("Add source documents or replace the ground-truth question file without leaving the Knowledge Base section.")
        render_quick_uploads("knowledge_base", rerun_after_upload=True)
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



def _flatten_mapping(mapping: dict, prefix: str = "") -> dict:
    flattened={}
    for key,value in (mapping or {}).items():
        column=f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_mapping(value, column))
        else:
            flattened[column]=value
    return flattened

def _run_summary_rows(run_payload: dict) -> list[dict]:
    summary=run_payload.get("summary", {}) or {}
    aggregate=summary.get("aggregate", {}) or {}
    return [{"metric":key,"value":value} for key,value in _flatten_mapping(aggregate).items()]

def _results_comparison_frame(run_results: dict[str, list[dict]]) -> pd.DataFrame:
    comparison=None
    for run_id,results in run_results.items():
        rows=[]
        for item in results:
            row={"case_id":item.get("case_id", ""),"question":item.get("question", "")}
            for metric,value in (item.get("metrics", {}) or {}).items():
                row[f"{run_id} · {metric}"]=value
            rows.append(row)
        run_frame=pd.DataFrame(rows)
        if run_frame.empty:
            run_frame=pd.DataFrame(columns=["case_id","question"])
        comparison=run_frame if comparison is None else comparison.merge(run_frame, on=["case_id","question"], how="outer")
    return comparison if comparison is not None else pd.DataFrame()

def _add_two_run_deltas(frame: pd.DataFrame, first_run_id: str, second_run_id: str) -> pd.DataFrame:
    delta_frame=frame.copy()
    first_prefix=f"{first_run_id} · "
    second_prefix=f"{second_run_id} · "
    for first_column in [column for column in frame.columns if column.startswith(first_prefix)]:
        metric=first_column[len(first_prefix):]
        second_column=f"{second_prefix}{metric}"
        if second_column not in frame.columns:
            continue
        first_values=pd.to_numeric(frame[first_column], errors="coerce")
        second_values=pd.to_numeric(frame[second_column], errors="coerce")
        if first_values.notna().any() or second_values.notna().any():
            delta_frame[f"Δ {metric} ({second_run_id} - {first_run_id})"]=second_values - first_values
    return delta_frame

def show_run_comparison(runs: list[dict]):
    st.subheader("Compare benchmark runs")
    run_ids=[run["id"] for run in runs]
    default_run_ids=run_ids[:2] if len(run_ids) >= 2 else run_ids
    selected_run_ids=st.multiselect(
        "Choose two or more run IDs to compare",
        run_ids,
        default=default_run_ids,
    )
    if len(selected_run_ids) < 2:
        st.info("Select at least two runs to build a comparison.")
        return

    run_payloads={}
    run_results={}
    try:
        for selected_run_id in selected_run_ids:
            run_payloads[selected_run_id]=requests.get(f"{API}/benchmarks/{selected_run_id}", timeout=30).json()
            run_results[selected_run_id]=requests.get(f"{API}/benchmarks/{selected_run_id}/results", timeout=60).json()
    except Exception as e:
        st.error(f"Could not load comparison data: {e}")
        return

    st.write("Run summaries")
    for column,selected_run_id in zip(st.columns(len(selected_run_ids)), selected_run_ids):
        with column:
            payload=run_payloads[selected_run_id]
            st.markdown(f"**{selected_run_id}**")
            st.caption(f"{payload.get('name', 'benchmark')} · {payload.get('status', 'unknown status')}")
            summary_rows=_run_summary_rows(payload)
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No aggregate summary metrics available.")

    comparison_frame=_results_comparison_frame(run_results)
    if comparison_frame.empty:
        st.info("No scored result metrics are available for the selected runs yet.")
        return
    if len(selected_run_ids) == 2:
        comparison_frame=_add_two_run_deltas(comparison_frame, selected_run_ids[0], selected_run_ids[1])
    st.write("Joined result metrics by case and question")
    st.dataframe(comparison_frame, use_container_width=True)

def show_results_layer():
    st.header("3. Benchmark Results")
    section_intro("Analyze run performance", "Review completed and queued runs, compare metrics, inspect traces, and open the raw JSON when deeper debugging is needed.")
    try:
        runs=requests.get(f"{API}/benchmarks", timeout=30).json()
        if runs:
            st.dataframe(pd.DataFrame(runs), use_container_width=True)
            run_id=st.selectbox("Choose a benchmark run", [r["id"] for r in runs])
            if run_id:
                summary_tab,results_tab,heatmap_tab,comparison_tab,raw_tab=st.tabs(["📊 Run summary", "🏁 Scored results", "🔥 Retrieval heatmaps", "🔀 Run comparison", "🧾 Raw JSON"])
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
                with heatmap_tab:
                    results=requests.get(f"{API}/benchmarks/{run_id}/results", timeout=60).json()
                    show_retrieval_heatmaps(results)
                with comparison_tab:
                    show_run_comparison(runs)
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
