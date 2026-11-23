import type {
  AcceptanceDecision,
  RecommendationMeasurements,
} from "../components/recommendation-evidence";

export function measuredFixture(): RecommendationMeasurements {
  return {
    version: 1,
    baseline_trial_id: "holdout-baseline",
    candidate_trial_id: "holdout-candidate",
    metrics: [
      {
        metric: "answer_correctness",
        label: "Correctness score",
        unit: "score",
        aggregation: "paired_mean",
        baseline: 0.5,
        candidate: 0.55,
        delta: 0.05,
        ci95: [-0.02, 0.12],
        paired_count: 6,
        baseline_reason: null,
        candidate_reason: null,
      },
      {
        metric: "claim_support",
        label: "Claim support score",
        unit: "score",
        aggregation: "paired_mean",
        baseline: 0.7,
        candidate: 0.8,
        delta: 0.1,
        ci95: [0.01, 0.2],
        paired_count: 6,
        baseline_reason: null,
        candidate_reason: null,
      },
      {
        metric: "citation_support",
        label: "Citation support score",
        unit: "score",
        aggregation: "paired_mean",
        baseline: null,
        candidate: null,
        delta: null,
        ci95: null,
        paired_count: 0,
        baseline_reason: "paired scores unavailable",
        candidate_reason: "paired scores unavailable",
      },
      {
        metric: "p95_seconds",
        label: "p95 query latency",
        unit: "seconds",
        aggregation: "p95",
        baseline: 1.2,
        candidate: 0.8,
        delta: -0.4,
        ci95: null,
        paired_count: null,
        baseline_reason: null,
        candidate_reason: null,
      },
      {
        metric: "cost_per_query",
        label: "Estimated cost per query",
        unit: "USD/query",
        aggregation: "mean",
        baseline: 0,
        candidate: null,
        delta: null,
        ci95: null,
        paired_count: null,
        baseline_reason: null,
        candidate_reason: "unknown pricing",
      },
    ],
    coverage: { paired: 6, expected: 6, fraction: 1, required_fraction: 0.95 },
    check_details: {
      statistical_confirmation:
        "Lower bound of the 95% correctness interval: -0.02; required: > 0.",
    },
    configuration_differences: [
      { setting: "retriever.type", baseline: "dense", candidate: "bm25" },
    ],
    evaluation: {
      synthetic_data: true,
      dataset_provenance: {
        status: "generated",
        dataset_id: "dataset-1",
        source_dataset_id: "dataset-1",
        source_session_id: "session",
        question_count: 30,
      },
      split_counts: { tuning: 24, holdout: 6 },
      human_reviewed_questions: 2,
      confirmation_stop_reason: null,
    },
  };
}

export function measuredAcceptance(): AcceptanceDecision {
  return {
    outcome: "promising_candidate",
    recommended_source: "candidate",
    reason_codes: ["statistical_confirmation"],
    checks: [
      {
        code: "paired_coverage",
        label: "At least 95% paired scores",
        status: "pass",
        measured: { paired: 6, expected: 6, fraction: 1 },
        threshold: 0.95,
        required: true,
      },
      {
        code: "statistical_confirmation",
        label: "Improvement is statistically confirmed",
        status: "fail",
        measured: -0.02,
        threshold: 0,
        required: false,
      },
    ],
  };
}
