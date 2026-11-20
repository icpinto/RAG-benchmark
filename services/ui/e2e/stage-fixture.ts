import type {
  StageDashboard,
  StageTrial,
} from "../src/components/optimization-stages";

export function stageFixture(
  trials: {
    id: string;
    phase: "retrieval" | "answer" | "confirm";
    status: string;
  }[] = [],
): StageDashboard {
  const metric = (mean: number | null, reason: string | null = null) => ({
    mean,
    reason,
    valid_count: mean == null ? 0 : 24,
    missing_count: mean == null ? 24 : 0,
  });
  const trial = (item: (typeof trials)[number]): StageTrial => ({
    ...item,
    split: item.phase === "confirm" ? "holdout" : "tuning",
    display_status: item.status,
    metrics: {
      "reranked_ndcg@10": metric(0.8),
      "raw_retrieval_ndcg@10": metric(0.5),
      answer_correctness: metric(0.9),
      claim_support: metric(0.85),
      citation_support: metric(0.8),
      total_estimated_cost: metric(null, "unknown pricing"),
      context_tokens: metric(100),
      query_latency_seconds: metric(0.1),
    },
    ranking:
      item.status === "completed"
        ? {
            rank: 1,
            score: item.phase === "retrieval" ? 0.8 : 0.9,
            paired_count: 24,
            missing_count: 0,
            feasible: true,
            violations: [],
          }
        : null,
    score_reason: item.status === "completed" ? null : "pending",
    p95_seconds: 0.1,
    coverage: { expected: 24, valid: 24, missing: 0, failed: 0 },
    configuration: {
      chunker: { type: "recursive", chunk_size: 512, overlap: 0.1 },
      chunk_units: "characters",
      embedding: { model: "saved-embedding" },
      indexers: { dense: { strategy: "dense_hnsw" } },
      retriever: { type: "dense" },
      rewriter: { type: "none" },
      reranker: { type: "none" },
      compressor: { type: "top_n", max_chunks: 5 },
      context: { max_context_chars: 6000 },
      prompt: { preset_id: "concise_v1" },
      llm: { model: "fixed-answer" },
    },
    roles: [
      item.phase === "retrieval"
        ? "shortlisted"
        : item.phase === "answer"
          ? "selected for holdout"
          : "holdout candidate",
    ],
    selection_reasons: ["Best feasible observed configuration"],
    source_trial_id:
      item.phase === "retrieval"
        ? null
        : (trials.find((t) => t.phase === "retrieval")?.id ?? null),
    source_answer_trial_id:
      item.phase === "confirm"
        ? (trials.find((t) => t.phase === "answer")?.id ?? null)
        : null,
    changes_from_baseline: [],
  });
  return {
    session_id: "session",
    workflow_version: 2,
    stages: {
      retrieval: {
        status: "completed",
        split: "tuning",
        question_count: 24,
        primary_metric: "reranked_ndcg@10",
        trials: trials.filter((t) => t.phase === "retrieval").map(trial),
      },
      answer: {
        status: "completed",
        split: "tuning",
        question_count: 24,
        primary_metric: "answer_correctness",
        trials: trials.filter((t) => t.phase === "answer").map(trial),
        plan: [
          {
            variant: "concise_v1",
            candidate_hash: "concise",
            source_trial_id:
              trials.find((t) => t.phase === "retrieval")?.id ?? null,
            scheduled: true,
            trial_id: trials.find((t) => t.phase === "answer")?.id ?? null,
            status: "completed",
          },
          {
            variant: "reranker:cross_encoder",
            candidate_hash: "reranker",
            source_trial_id: null,
            scheduled: false,
            trial_id: null,
            status: "outside trial cap",
          },
        ],
      },
      confirm: {
        status: "completed",
        split: "holdout",
        question_count: 6,
        primary_metric: "answer_correctness",
        trials: trials.filter((t) => t.phase === "confirm").map(trial),
        outcome: "confirmed improvement",
        baseline_retained: false,
        selected_trial_id: trials.find((t) => t.phase === "answer")?.id ?? null,
        comparison: {
          compatible: true,
          metrics: [
            {
              metric: "answer_correctness",
              baseline_mean: 0.5,
              candidate_mean: 0.9,
              mean_delta: 0.4,
              ci95: [0.2, 0.6],
              paired_count: 6,
              missing_count: 0,
            },
          ],
        },
        configuration_differences: [
          {
            setting: "prompt.preset_id",
            baseline: "baseline_v1",
            candidate: "concise_v1",
          },
        ],
      },
    },
  };
}
