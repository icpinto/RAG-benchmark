"""Version-five stage boundaries and compatibility with saved optimizer sessions."""

from copy import deepcopy

import pytest
from langgraph.types import Command
from ragbench_core.optimization import adaptive, memory, store
from ragbench_core.optimization.candidates import answer_plan, retrieval_identity
from ragbench_core.optimization.contracts import TrialRequest
from ragbench_core.optimization.selection import compare_trials
from ragbench_core.optimization.stages import stage_dashboard
from ragbench_core.optimization.workflow import progress
from ragbench_core.prompt_optimization.auto import plan_answers
from test_adaptive_optimization import (
    choose,
    complete,
    graph_for,
    stop,
    tools,  # noqa: F401 -- pytest fixture
    transport,
)
from test_optimization import database, session, template  # noqa: F401


@pytest.fixture
def raw_tools(request):
    optimizer = request.getfixturevalue("tools")
    progress(optimizer.session_id, workflow_version=5)
    run_template = deepcopy(optimizer.session["template"])
    run_template["pipeline"]["query"]["rerankers"] = [
        {"type": "diversity_mmr", "lambda_mult": 0.7, "top_n": 8}
    ]
    run_request = deepcopy(optimizer.session["request"])
    run_request["allowed_components"] = {"rerankers": ["diversity_mmr", "flashrank"]}
    store.update(optimizer.session_id, template=run_template, request=run_request)
    optimizer.registry.rerankers.update(diversity_mmr=object(), flashrank=object())
    return optimizer


def test_retrieval_control_and_seed_suggestions_disable_reranking(raw_tools):
    baseline = raw_tools.baseline()
    adaptive.initialize(raw_tools, "retrieval")
    action = raw_tools.read(adaptive.phase_state(raw_tools, "retrieval")["pending"])
    assert action["config"]["pipeline"]["query"]["rerankers"] == [{"type": "none"}]
    assert raw_tools.session["progress"]["baseline_config"] == baseline
    assert raw_tools.baseline() == baseline
    assert raw_tools.get_search_space("retrieval")["rerankers"] == ["none"]
    assert raw_tools.get_search_space()["rerankers"] == ["flashrank", "diversity_mmr"]
    seeds = adaptive.available_candidates(raw_tools, "retrieval")
    assert seeds
    assert all(c["configuration"]["reranker"] == {"type": "none"} for c in seeds)
    with pytest.raises(ValueError, match="outside approved search space"):
        raw_tools.validate_candidate(action["config"])


def test_active_reranker_rejected_before_retrieval_dispatch(raw_tools):
    dispatched = []
    for reranker in (
        raw_tools.baseline()["pipeline"]["query"]["rerankers"],
        [{"type": "flashrank"}],
    ):
        config = raw_tools.retrieval_config()
        config["pipeline"]["query"]["rerankers"] = reranker
        with pytest.raises(ValueError):
            adaptive.validate_decision(
                raw_tools,
                "retrieval",
                {**choose({}, candidate_id="unlisted"), "pipeline": config["pipeline"]},
                False,
            )
        with pytest.raises(ValueError):
            raw_tools.run_trial(
                TrialRequest(
                    session_id=raw_tools.session_id, phase="retrieval", config=config
                ),
                dispatched.append,
            )
    assert not dispatched
    assert not store.rows(store.OptimizationTrial, raw_tools.session_id)


@pytest.mark.parametrize("winner", [True, False])
@pytest.mark.parametrize("maximum", [1, 20])
def test_answer_control_restores_template_reranker_and_source(
    raw_tools, winner, maximum
):
    baseline = raw_tools.baseline()
    retrieval = raw_tools.retrieval_config()
    if winner:
        retrieval["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    trials = [{"id": "retrieval", "phase": "retrieval", "config": retrieval}]
    ranking = [{"trial_id": "retrieval", "feasible": winner}]
    plan = answer_plan(raw_tools, trials, ranking, maximum)
    control = plan["answer_candidates"][0]
    assert (
        control["pipeline"]["query"]["rerankers"]
        == baseline["pipeline"]["query"]["rerankers"]
    )
    assert retrieval_identity(control) == retrieval_identity(retrieval)
    assert all(e["source_trial_id"] == "retrieval" for e in plan["answer_plan"])
    assert len(plan["answer_candidates"]) <= maximum
    assert retrieval["pipeline"]["query"]["rerankers"] == [{"type": "none"}]
    assert any(e["variant"] == "reranker:flashrank" for e in plan["answer_plan"])
    assert all(
        c["pipeline"]["query"]["rerankers"][0]["type"] != "none"
        for c in plan["answer_candidates"]
    )
    if not winner:
        assert control == baseline
        assert not plan["retrieval_shortlist"]


def test_raw_ranking_uses_raw_ndcg_and_recall_with_legacy_defaults():
    ids = ["question"]
    trials = [{"id": name} for name in ("raw-best", "recall-best", "reranked-best")]
    rows = {
        name: [
            {
                "question_id": "question",
                "metrics": {
                    "raw_retrieval_ndcg@10": ndcg,
                    "raw_retrieval_recall@10": recall,
                    "reranked_ndcg@10": reranked,
                    "reranked_recall@10": 1 - recall,
                },
            }
        ]
        for name, ndcg, recall, reranked in (
            ("raw-best", 0.9, 0.6, 0.2),
            ("recall-best", 0.9, 0.8, 0.1),
            ("reranked-best", 0.4, 1, 1),
        )
    }
    ranked = compare_trials(trials, rows, ids, "retrieval", {}, workflow_version=5)
    assert [r["trial_id"] for r in ranked] == [
        "recall-best",
        "raw-best",
        "reranked-best",
    ]
    assert all(r["metric"] == "raw_retrieval_ndcg@10" for r in ranked)
    assert ranked[0]["recall"] == 0.8
    for version in (1, 2, 3, 4):
        legacy = compare_trials(
            trials, rows, ids, "retrieval", {}, workflow_version=version
        )
        assert legacy[0]["trial_id"] == "reranked-best"
    assert (
        compare_trials(trials, rows, ids, "retrieval", {})[0]["metric"]
        == "reranked_ndcg@10"
    )


def test_dspy_control_restores_template_reranker(raw_tools):
    request = deepcopy(raw_tools.session["request"])
    request["prompt_optimization"] = {"configs": [{"preset": "mipro_zero_shot_v1"}]}
    store.update(raw_tools.session_id, request=request)
    retrieval = raw_tools.retrieval_config()
    retrieval["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    with store.db.SessionLocal.begin() as conn:
        conn.add(
            store.OptimizationTrial(
                id="winner",
                session_id=raw_tools.session_id,
                operation_key="winner",
                phase="retrieval",
                split="tuning",
                status="completed",
                config=retrieval,
            )
        )
    progress(
        raw_tools.session_id,
        retrieval_ranking=[{"trial_id": "winner", "feasible": True}],
    )
    plan = plan_answers(raw_tools, 1)
    control, compiled = plan["prompt_answer_plan"][:2]
    assert control["scheduled"] and not compiled["scheduled"]
    assert compiled["variant"] == "dspy:mipro_zero_shot_v1"
    assert control["config"] == compiled["config"]
    assert (
        control["config"]["pipeline"]["query"]["rerankers"]
        == raw_tools.baseline()["pipeline"]["query"]["rerankers"]
    )
    assert retrieval_identity(control["config"]) == retrieval_identity(retrieval)
    assert control["source_trial_id"] == compiled["source_trial_id"] == "winner"


def test_v5_resume_context_dashboard_and_exact_confirmation(raw_tools):
    dispatched, calls = [], []
    raw_tools.model_call = transport(stop, calls)
    graph, config = graph_for(raw_tools, dispatched.append)
    graph.invoke({"session_id": raw_tools.session_id, "stage": "retrieval"}, config)
    pending = adaptive.phase_state(raw_tools, "retrieval")["pending"]
    graph.invoke(Command(resume=True), config)
    assert len(dispatched) == 1
    assert adaptive.phase_state(raw_tools, "retrieval")["pending"] == pending
    complete(raw_tools, dispatched[0])
    graph.invoke(Command(resume=True), config)
    context = calls[0]
    assert context["objective"]["metric"] == "raw_retrieval_ndcg@10"
    assert context["current_observation"]["metric"] == "raw_retrieval_ndcg@10"
    assert context["search_space"]["rerankers"] == ["none"]
    assert context["baseline_pipeline"]["query"]["rerankers"] == [{"type": "none"}]
    assert "Reranking is disabled" in context["rules"]["fixed"]
    dashboard = stage_dashboard(raw_tools.session)
    retrieval = dashboard["stages"]["retrieval"]
    assert dashboard["workflow_version"] == 5
    assert retrieval["primary_metric"] == "raw_retrieval_ndcg@10"
    assert retrieval["trials"][0]["ranking"]["score"] == 0.5
    assert "baseline" in retrieval["trials"][0]["roles"]
    answer = store.rows(store.OptimizationTrial, raw_tools.session_id)[1]
    assert answer["phase"] == "answer"
    assert (
        answer["config"]["pipeline"]["query"]["rerankers"]
        == raw_tools.baseline()["pipeline"]["query"]["rerankers"]
    )
    # Finish each queued trial across recovery boundaries, including holdout.
    for _ in range(4):
        if raw_tools.session["status"] == "completed":
            break
        complete(raw_tools, dispatched[-1])
        graph.invoke(Command(resume=True), config)
    assert raw_tools.session["status"] == "completed"
    confirmation = [
        t
        for t in store.rows(store.OptimizationTrial, raw_tools.session_id)
        if t["phase"] == "confirm"
    ]
    assert confirmation
    assert all(
        t["config"]["pipeline"]["query"]["rerankers"]
        == raw_tools.baseline()["pipeline"]["query"]["rerankers"]
        for t in confirmation
    )


def test_retrieval_observation_reads_raw_evidence_only(raw_tools):
    path = raw_tools.write(
        "trace.json", {"raw": ["raw evidence"], "reranked": ["reranked evidence"]}
    )
    row = {"artifact_paths": {"retrieval_traces": str(raw_tools.root / path)}}
    assert (
        memory.evidence_from_artifact(raw_tools, row, "retrieval") == '["raw evidence"]'
    )
    assert (
        memory.evidence_from_artifact(raw_tools, row, "answer")
        == '["reranked evidence"]'
    )


def test_adaptive_answering_inherits_raw_winner_when_reranked_scores_disagree(
    raw_tools,
):
    calls = []

    def decide(data):
        if data["objective"]["phase"] == "retrieval" and data["remaining"]["trials"]:
            candidate = next(
                c
                for c in data["candidates"]
                if c["configuration"]["retriever"] == {"type": "bm25"}
            )
            return choose(data, candidate_id=candidate["candidate_id"])
        return stop(data)

    raw_tools.model_call = transport(decide, calls)

    def dispatch(run_id):
        with store.db.SessionLocal() as conn:
            trial = conn.get(store.OptimizationTrial, run_id)
            score = (
                0.9
                if trial.config["pipeline"]["query"]["retrievers"] == [{"type": "bm25"}]
                else 0.2
            )
            phase = trial.phase
        complete(raw_tools, run_id, score)
        if phase == "retrieval":
            with store.db.SessionLocal.begin() as conn:
                for row in conn.execute(
                    store.select(store.db.BenchmarkResult).where(
                        store.db.BenchmarkResult.run_id == run_id
                    )
                ).scalars():
                    row.metrics = {**row.metrics, "reranked_ndcg@10": 1 - score}

    graph, config = graph_for(raw_tools, dispatch)
    graph.invoke({"session_id": raw_tools.session_id, "stage": "retrieval"}, config)
    assert raw_tools.session["status"] == "completed"
    assert calls[1]["current_observation"]["comparisons"]["initial"][
        "delta"
    ] == pytest.approx(0.7)
    winner = raw_tools.session["progress"]["retrieval_ranking"][0]
    assert winner["score"] == pytest.approx(0.9)
    trials = store.rows(store.OptimizationTrial, raw_tools.session_id)
    answer = next(t for t in trials if t["phase"] == "answer")
    assert answer["config"]["pipeline"]["query"]["retrievers"] == [{"type": "bm25"}]
    assert (
        answer["config"]["pipeline"]["query"]["rerankers"]
        == raw_tools.baseline()["pipeline"]["query"]["rerankers"]
    )
    dashboard = stage_dashboard(raw_tools.session)
    assert (
        dashboard["stages"]["answer"]["trials"][0]["source_trial_id"]
        == winner["trial_id"]
    )
    scores = [t["ranking"]["score"] for t in dashboard["stages"]["retrieval"]["trials"]]
    assert sorted(scores) == pytest.approx([0.2, 0.9])


def test_v4_preserves_template_reranking_and_legacy_ranking_policy(raw_tools):
    progress(raw_tools.session_id, workflow_version=4)
    adaptive.initialize(raw_tools, "retrieval")
    action = raw_tools.read(adaptive.phase_state(raw_tools, "retrieval")["pending"])
    assert action["config"] == raw_tools.baseline()
    assert raw_tools.get_search_space("retrieval")["rerankers"] == [
        "flashrank",
        "diversity_mmr",
    ]
    assert all(
        c["pipeline"]["query"]["rerankers"]
        == raw_tools.baseline()["pipeline"]["query"]["rerankers"]
        for c in adaptive.available_candidates(raw_tools, "retrieval")
    )
    assert (
        stage_dashboard(raw_tools.session)["stages"]["retrieval"]["primary_metric"]
        == "reranked_ndcg@10"
    )
