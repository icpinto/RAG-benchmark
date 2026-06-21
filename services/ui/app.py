from __future__ import annotations
import os
import html
import requests
import pandas as pd
import streamlit as st

API=os.getenv("API_BASE_URL", "http://api:8000")


STATUS_BADGE_STYLES={
    "queued": ("#92400E", "#FEF3C7", "#F59E0B", "⏳"),
    "running": ("#1D4ED8", "#DBEAFE", "#3B82F6", "🔄"),
    "completed": ("#047857", "#D1FAE5", "#10B981", "✅"),
    "failed": ("#B91C1C", "#FEE2E2", "#EF4444", "❌"),
}

def _normalize_run_status(status: str | None) -> str:
    return (status or "unknown").strip().lower()

def _status_badge_html(status: str | None) -> str:
    normalized=_normalize_run_status(status)
    color,background,border,icon=STATUS_BADGE_STYLES.get(
        normalized,
        ("#374151", "#F3F4F6", "#9CA3AF", "ℹ️"),
    )
    label=html.escape(normalized.title())
    return (
        f'<span class="run-status-badge" style="color:{color}; background:{background}; '
        f'border-color:{border};">{icon} {label}</span>'
    )

def _status_label(status: str | None) -> str:
    normalized=_normalize_run_status(status)
    icon=STATUS_BADGE_STYLES.get(normalized, STATUS_BADGE_STYLES["queued"])[3] if normalized in STATUS_BADGE_STYLES else "ℹ️"
    return f"{icon} {normalized}"

def _runs_display_frame(runs: list[dict]) -> pd.DataFrame:
    frame=pd.DataFrame(runs)
    if frame.empty or "status" not in frame.columns:
        return frame
    frame=frame.copy()
    frame.insert(0, "status_label", frame["status"].map(_status_label))
    return frame

def _selected_run_guidance(status: str | None, results: list[dict] | None = None):
    normalized=_normalize_run_status(status)
    has_results=bool(results)
    if normalized == "queued":
        st.info("This run is queued and has not produced results yet. Use **Refresh runs** to check progress. If it remains queued for a long time, verify that the Celery worker is online and consuming the `ragbench` queue.")
    elif normalized == "running":
        st.info("This run is currently running. Results may appear incrementally; refresh this view in a moment to load the latest output.")
    elif normalized == "failed":
        st.error("This run failed. Open the raw run summary for error details, then check API/worker logs and the benchmark configuration before re-running.")
    elif normalized == "completed" and not has_results:
        st.warning("This run is marked completed but no scored results were returned. Check the artifact directory and worker logs for missing or empty result files.")
    elif not has_results:
        st.info("No results are available for this run yet. Refresh later or inspect the run summary for current status.")


def _response_detail(response: requests.Response | None, max_length: int = 500) -> str:
    if response is None:
        return ""
    text=(response.text or "").strip()
    if not text:
        return ""
    if len(text)>max_length:
        text=f"{text[:max_length]}…"
    return f" Response body: {text}"

def api_request_json(method: str, url: str, *, error_context: str, **kwargs):
    try:
        response=requests.request(method, url, **kwargs)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            st.error(f"{error_context}: API returned a non-JSON response.{_response_detail(response)}")
            return None
    except requests.RequestException as e:
        response=getattr(e, "response", None)
        st.error(f"{error_context}: {e}.{_response_detail(response)}")
        return None

def api_get_json(url: str, *, error_context: str, **kwargs):
    return api_request_json("GET", url, error_context=error_context, **kwargs)

def api_post_json(url: str, *, error_context: str, **kwargs):
    return api_request_json("POST", url, error_context=error_context, **kwargs)

def api_put_json(url: str, *, error_context: str, **kwargs):
    return api_request_json("PUT", url, error_context=error_context, **kwargs)

def api_delete_json(url: str, *, error_context: str, **kwargs):
    return api_request_json("DELETE", url, error_context=error_context, **kwargs)


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

        .run-status-badge {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            border: 1px solid;
            border-radius: 999px;
            padding: .2rem .65rem;
            font-weight: 700;
            font-size: .9rem;
            line-height: 1.4;
            white-space: nowrap;
        }
        .run-guidance-card {
            border: 1px solid #D7E3F4;
            border-radius: .9rem;
            background: #F8FBFF;
            padding: .9rem 1rem;
            margin: .6rem 0 1rem 0;
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

            payload=api_post_json(f"{API}/data/upload", files=files, timeout=120, error_context="Could not upload source documents")
            if payload is not None:
                st.success(payload)
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

            payload=api_post_json(f"{API}/data/ground-truth", files={"file":("qa.json", gt.getvalue())}, timeout=120, error_context="Could not upload ground-truth JSON")
            if payload is not None:
                st.success(payload)
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

            listing=api_get_json(f"{API}/data/files", timeout=30, error_context="Could not load data files")
            if listing is None:
                return
            files=listing.get("files", [])
            st.caption(f"Raw data directory: {listing.get('data_root', 'unknown')}")
            if files:
                st.dataframe(pd.DataFrame(files), use_container_width=True)
                selected_file=st.selectbox("Choose a source file to preview", [f["path"] for f in files])
                if selected_file:
                    try:

                        content=api_get_json(f"{API}/data/files/content", params={"path":selected_file}, timeout=30, error_context="Could not preview file")
                        if content is not None:
                            st.text_area("Source file preview", value=content.get("content", ""), height=360, disabled=True)
                    except Exception as e:
                        st.warning(f"Could not preview file: {e}")
            else:
                st.info("No raw data files have been uploaded yet.")
        except Exception as e:
            st.error(f"Could not load data files: {e}")
    with questions_tab:
        try:

            payload=api_get_json(f"{API}/data/questions", timeout=30, error_context="Could not load evaluation questions")
            if payload is None:
                return
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

                    payload=api_put_json(f"{API}/data/questions/{selected_index}", json=edited, timeout=30, error_context="Could not save question")
                    if payload is not None:
                        st.success("Question saved")
                        st.rerun()
                if delete_clicked:

                    payload=api_delete_json(f"{API}/data/questions/{selected_index}", timeout=30, error_context="Could not delete question")
                    if payload is not None:
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

                        payload=api_post_json(f"{API}/data/questions", json=new_question, timeout=30, error_context="Could not add question")
                        if payload is not None:
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

        payload=api_get_json(f"{API}/configs/content", params={"config_path":path}, timeout=30, error_context="Could not load config")
        if payload is None:
            return False
        st.session_state.config_path=payload["config_path"]
        if sync_path_input:
            st.session_state.config_path_input=payload["config_path"]
        st.session_state.config_editor_content=payload["content"]
        st.session_state.config_editor_text=payload["content"]

        st.session_state.config_dirty=False
        return True

    try:

        config_listing=api_get_json(f"{API}/configs", timeout=30, error_context="Could not load saved configurations") or {"config_root":"/app/configs","files":[]}
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

                    if load_config(selected_config, sync_path_input=True):
                        st.success(f"Loaded {st.session_state.config_path}")
                except Exception as e:
                    st.error(f"Could not load config: {e}")
        else:
            st.info("No saved configs were found.")

        config_path=st.text_input("Config path in container", key="config_path_input")
        st.session_state.config_path=config_path
        if st.button("Load from path"):
            try:

                if load_config(st.session_state.config_path):
                    st.success(f"Loaded {st.session_state.config_path}")
            except Exception as e:
                st.error(f"Could not load config: {e}")

        st.divider()
        st.subheader("Launch benchmark")
        selected_source_paths=[]
        selected_question_indices=[]
        try:

            files_payload=api_get_json(f"{API}/data/files", timeout=30, error_context="Could not load source document choices")
            if files_payload is None:
                files_payload={"files":[]}
            source_files=files_payload.get("files", [])
            source_options=[item.get("path", "") for item in source_files if item.get("path")]
            selected_source_paths=st.multiselect(
                "Source documents to include",
                source_options,
                default=source_options,
                help="Leave all selected to benchmark every uploaded source document. Clear selections to run with no source documents.",
            )
        except Exception as e:
            st.warning(f"Could not load source document choices: {e}")
        try:

            questions_payload=api_get_json(f"{API}/data/questions", timeout=30, error_context="Could not load evaluation question choices")
            if questions_payload is None:
                questions_payload={"questions":[]}
            questions=questions_payload.get("questions", [])
            question_options=list(range(len(questions)))
            selected_question_indices=st.multiselect(
                "Evaluation questions to include",
                question_options,
                default=question_options,
                format_func=lambda index: _short_question_label(questions[index].get("question", ""), index),
                help="Leave all selected to benchmark every ground-truth question. Clear selections to run with no questions.",
            )
        except Exception as e:
            st.warning(f"Could not load evaluation question choices: {e}")
        if st.button("Start benchmark run"):
            if st.session_state.config_dirty:
                st.warning("Unsaved YAML edits were not used. Click Save YAML first to benchmark the edited config.")
            payload={
                "config_path":st.session_state.config_path,
                "source_paths":selected_source_paths,
                "question_indices":selected_question_indices,
            }

            run_payload=api_post_json(f"{API}/benchmarks/run", json=payload, timeout=30, error_context="Could not start benchmark run")
            if run_payload is not None:
                st.success(run_payload)

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

                payload=api_put_json(
                    f"{API}/configs/content",
                    json={"config_path":st.session_state.config_path,"content":edited_content},
                    timeout=30,
                    error_context="Could not save config",
                )
                if payload is None:
                    return
                st.session_state.config_path=payload["config_path"]
                st.session_state.config_editor_content=edited_content
                st.session_state.config_dirty=False
                st.success(f"Saved {payload['config_path']}")
            except Exception as e:
                st.error(f"Could not save config: {e}")
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
            run_payload=api_get_json(f"{API}/benchmarks/{selected_run_id}", timeout=30, error_context=f"Could not load run {selected_run_id}")
            results_payload=api_get_json(f"{API}/benchmarks/{selected_run_id}/results", timeout=60, error_context=f"Could not load results for run {selected_run_id}")
            if run_payload is None or results_payload is None:
                return
            run_payloads[selected_run_id]=run_payload
            run_results[selected_run_id]=results_payload
    except Exception as e:
        st.error(f"Could not load comparison data: {e}")
        return

    st.write("Run summaries")
    for column,selected_run_id in zip(st.columns(len(selected_run_ids)), selected_run_ids):
        with column:
            payload=run_payloads[selected_run_id]
            st.markdown(f"**{selected_run_id}**")
            st.caption(f"{payload.get('name', 'benchmark')} · {payload.get('status', 'unknown status')}")
            st.write({
                "config_hash": payload.get("config_hash"),
                "dataset_hash": payload.get("dataset_hash"),
                "models": payload.get("model_identifiers", {}),
            })
            summary_rows=_run_summary_rows(payload)
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No aggregate summary metrics available.")

    metadata_rows=[]
    for selected_run_id,payload in run_payloads.items():
        metadata_rows.append({
            "run_id": selected_run_id,
            "config_hash": payload.get("config_hash"),
            "dataset_hash": payload.get("dataset_hash"),
            "models": payload.get("model_identifiers", {}),
            "artifact_dir": payload.get("artifact_dir"),
        })
    if metadata_rows:
        st.write("Run metadata for grouping by config, dataset, and model")
        st.dataframe(pd.DataFrame(metadata_rows), use_container_width=True)

    comparison_frame=_results_comparison_frame(run_results)
    if comparison_frame.empty:
        st.info("No scored result metrics are available for the selected runs yet.")
        return
    if len(selected_run_ids) == 2:
        comparison_frame=_add_two_run_deltas(comparison_frame, selected_run_ids[0], selected_run_ids[1])
    grouping=st.multiselect("Comparison grouping columns", ["case_id","question","config_hash","dataset_hash","models"], default=["case_id","question"])
    if any(column in grouping for column in ["config_hash","dataset_hash","models"]):
        st.caption("Config, dataset, and model identifiers are shown in the metadata table above; metric deltas remain joined by case and question.")
    st.write("Joined result metrics and metric deltas by case and question")
    st.dataframe(comparison_frame, use_container_width=True)

def show_results_layer():
    st.header("3. Benchmark Results")
    section_intro("Analyze run performance", "Review completed and queued runs, compare metrics, inspect traces, and open the raw JSON when deeper debugging is needed.")
    try:
        runs=api_get_json(f"{API}/benchmarks", timeout=30, error_context="API unavailable")
        if runs is None:
            return
        if runs:
            refresh_col,help_col=st.columns([1,3])
            with refresh_col:
                if st.button("Refresh runs", help="Reload benchmark run statuses and results from the API."):
                    st.rerun()
            with help_col:
                st.caption("Statuses are loaded from `/benchmarks`; select a run to load its latest summary from `/benchmarks/{run_id}`.")

            st.dataframe(_runs_display_frame(runs), use_container_width=True)
            run_id=st.selectbox("Choose a benchmark run", [r["id"] for r in runs])
            if run_id:
                selected_run=next((run for run in runs if run.get("id") == run_id), {})
                selected_status=selected_run.get("status")
                st.markdown(
                    f'<div class="run-guidance-card"><strong>Selected run:</strong> {html.escape(str(run_id))} '&
                    f'{_status_badge_html(selected_status)}</div>',
                    unsafe_allow_html=True,
                )
                summary_tab,results_tab,heatmap_tab,comparison_tab,raw_tab=st.tabs(["📊 Run summary", "🏁 Scored results", "🔥 Retrieval heatmaps", "🔀 Run comparison", "🧾 Raw JSON"])
                with summary_tab:
                    run_payload=api_get_json(f"{API}/benchmarks/{run_id}", timeout=30, error_context="Could not load run summary")
                    if run_payload is not None:
                        selected_status=run_payload.get("status", selected_status)
                        st.markdown(_status_badge_html(selected_status), unsafe_allow_html=True)
                        _selected_run_guidance(selected_status)
                        st.json(run_payload)
                with results_tab:
                    run_payload=api_get_json(f"{API}/benchmarks/{run_id}", timeout=30, error_context="Could not load run summary")
                    if run_payload is not None:
                        selected_status=run_payload.get("status", selected_status)
                    results=api_get_json(f"{API}/benchmarks/{run_id}/results", timeout=60, error_context="Could not load run results")
                    if results is None:
                        return
                    if results:
                        rows=[]
                        for item in results:
                            row={"case_id":item["case_id"],"question":item["question"],"answer":item["answer"]}
                            row.update(item.get("metrics",{}))
                            rows.append(row)
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                        show_evaluation_step_details(results)
                    else:
                        with st.spinner("Waiting for benchmark results…"):
                            _selected_run_guidance(selected_status, results)
                with heatmap_tab:
                    results=api_get_json(f"{API}/benchmarks/{run_id}/results", timeout=60, error_context="Could not load run results")
                    if results is None:
                        return
                    show_retrieval_heatmaps(results)
                with comparison_tab:
                    show_run_comparison(runs)
                with raw_tab:
                    results=api_get_json(f"{API}/benchmarks/{run_id}/results", timeout=60, error_context="Could not load run results")
                    if results is None:
                        return
                    if results:
                        st.json(results)
                    else:
                        st.info("No raw results are available for this run yet.")
        else:
            st.info("No benchmark runs yet.")
    except Exception as e:
        st.error(f"API unavailable: {e}")


navigation_options={
    "Knowledge Base":{
        "icon":"📚",
        "description":"Upload documents and manage ground truth",
        "page":"knowledge_base",
    },
    "Benchmark Setup":{
        "icon":"⚙️",
        "description":"Edit config and launch runs",
        "page":"benchmark_setup",
    },
    "Results Explorer":{
        "icon":"📊",
        "description":"Compare metrics and inspect traces",
        "page":"results_explorer",
    },
}
with st.sidebar:
    st.markdown("### 🧭 Workflow")
    selected_label=st.radio(
        "Choose a workspace",
        list(navigation_options),
        format_func=lambda label: f"{navigation_options[label]['icon']} {label}",
        label_visibility="collapsed",
    )
    st.caption(navigation_options[selected_label]["description"])

selected_page=navigation_options[selected_label]["page"]

if selected_page == "knowledge_base":
    show_data_library()
elif selected_page == "benchmark_setup":
    show_test_config_setup()
else:
    show_results_layer()
