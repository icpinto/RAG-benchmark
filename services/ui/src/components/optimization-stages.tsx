import { useState } from "react";
import { Link, useLocation, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Card, Empty, ErrorNotice, Loading, Notice } from "./common";
import { DataTable } from "./data-table";
import { TrialConfiguration } from "./optimization-config";
import {
  OptimizationHistory,
  type PlannerStatus,
} from "./optimization-history";
import {
  PromptCandidateHistory,
  type CompilationHistorySummary,
} from "./prompt-candidate-history";

type Phase = "retrieval" | "answer" | "confirm";
type Config = Record<string, unknown>;
interface Metric {
  mean: number | null;
  valid_count: number;
  missing_count: number;
  reason: string | null;
}
interface Ranking {
  rank: number;
  score: number;
  paired_count: number;
  missing_count: number;
  feasible: boolean;
  violations: string[];
}
export interface StageTrial {
  id: string;
  phase: Phase;
  split: string;
  status: string;
  display_status: string;
  metrics: Record<string, Metric>;
  ranking: Ranking | null;
  score_reason: string | null;
  p95_seconds: number | null;
  coverage: {
    expected: number;
    valid: number;
    missing: number;
    failed: number;
  };
  configuration: Config | null;
  roles: string[];
  selection_reasons: string[];
  source_trial_id: string | null;
  source_answer_trial_id: string | null;
  changes_from_baseline: Change[];
}
interface Change {
  setting: string;
  baseline: unknown;
  candidate: unknown;
}
interface ComparisonMetric {
  metric: string;
  baseline_mean: number | null;
  candidate_mean: number | null;
  mean_delta: number | null;
  ci95: number[] | null;
  paired_count: number;
  missing_count: number;
  reason?: string;
}
interface Planned {
  candidate_hash: string;
  source_trial_id: string | null;
  variant: string;
  reason?: string;
  scheduled: boolean;
  trial_id: string | null;
  status: string;
}
interface Stage {
  planner?: PlannerStatus;
  status: string;
  split: string;
  question_count: number;
  primary_metric: string;
  trials: StageTrial[];
  plan?: Planned[];
  compilations?: {
    id: string;
    status: string;
    progress: { current_config?: number };
    usage: { tokens: number; cost: number | null };
    history?: CompilationHistorySummary;
  }[];
  outcome?: string;
  selected_trial_id?: string | null;
  baseline_retained?: boolean;
  configuration_differences?: Change[];
  comparison?: {
    compatible: boolean;
    reason?: string;
    metrics: ComparisonMetric[];
  } | null;
}
export interface StageDashboard {
  session_id: string;
  workflow_version: number;
  stages: Record<Phase, Stage>;
}

const names: Record<Phase, string> = {
  retrieval: "Retrieval",
  answer: "Answering",
  confirm: "Confirmation",
};
const phases: Phase[] = ["retrieval", "answer", "confirm"];
const descriptions = {
  retrieval:
    "Find and rank relevant evidence. Higher NDCG, recall, precision and MRR are better; lower latency is better.",
  answer:
    "Evaluate complete answers on tuning questions. Higher semantic correctness, claim support and citation support are better; lower latency and cost are better.",
  confirm:
    "Compare the frozen candidate with the baseline on holdout questions excluded from tuning. A positive correctness delta favors the candidate.",
};
const fmt = (
  value: number | null | undefined,
  reason = "Not measured",
  digits = 3,
) => (value != null && Number.isFinite(value) ? value.toFixed(digits) : reason);
const describe = (value: unknown): string => {
  if (value == null) return "Not configured";
  if (Array.isArray(value)) return value.map(describe).join("; ");
  if (typeof value === "object")
    return Object.entries(value as Config)
      .map(([k, v]) => `${k}: ${describe(v)}`)
      .join(", ");
  return String(value);
};
const mean = (trial: StageTrial, key: string) => trial.metrics[key]?.mean;
const metricText = (trial: StageTrial, key: string) =>
  fmt(mean(trial, key), trial.metrics[key]?.reason ?? "Not measured");

function TrialLink({
  sessionId,
  trialId,
  phase,
}: {
  sessionId: string;
  trialId: string;
  phase: Phase;
}) {
  return (
    <Link to={`/auto-optimize/${sessionId}?stage=${phase}#trial-${trialId}`}>
      {trialId.slice(0, 8)}
    </Link>
  );
}

function Inspector({
  sessionId,
  trial,
  annotateSettings,
  workflowVersion,
}: {
  sessionId: string;
  trial: StageTrial;
  annotateSettings: boolean;
  workflowVersion: number;
}) {
  const [open, setOpen] = useState(false);
  const metrics = Object.entries(trial.metrics).map(([metric, value]) => ({
    metric,
    descriptive_mean: value.mean ?? value.reason ?? "Not measured",
    valid: value.valid_count,
    missing: value.missing_count,
    availability: value.reason ?? "Measured",
  }));
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>View config</summary>
      {open && (
        <div className="stage-inspector">
          <h3>Trial {trial.id}</h3>
          <p>{trial.selection_reasons.join(". ") || "Selection pending."}</p>
          {trial.phase !== "retrieval" && (
            <p>
              Source retrieval trial:{" "}
              {trial.source_trial_id ? (
                <TrialLink
                  sessionId={sessionId}
                  trialId={trial.source_trial_id}
                  phase="retrieval"
                />
              ) : (
                "Source trial unavailable"
              )}
              .
            </p>
          )}
          {trial.source_answer_trial_id && (
            <p>
              Selected answer trial:{" "}
              <TrialLink
                sessionId={sessionId}
                trialId={trial.source_answer_trial_id}
                phase="answer"
              />
              .
            </p>
          )}
          <TrialConfiguration
            sessionId={sessionId}
            trialId={trial.id}
            stage={annotateSettings ? trial.phase : undefined}
            workflowVersion={workflowVersion}
          />
          <h3>Measured metrics</h3>
          <p>
            Descriptive means use available observations. Selection scores use
            shared questions with at least 95% coverage.
          </p>
          <DataTable name={`metrics-${trial.id}`} rows={metrics} />
          {!!trial.changes_from_baseline.length && (
            <>
              <h3>Changes from baseline</h3>
              <DataTable
                name={`changes-${trial.id}`}
                rows={trial.changes_from_baseline.map((change) => ({
                  ...change,
                }))}
              />
            </>
          )}
        </div>
      )}
    </details>
  );
}

function ScoreBars({ trials, label }: { trials: StageTrial[]; label: string }) {
  const ranked = trials
    .filter((trial) => trial.ranking)
    .sort((a, b) => a.ranking!.rank - b.ranking!.rank);
  return (
    <figure className="stage-figure" aria-label={label}>
      <figcaption>{label} · higher is better</figcaption>
      {!ranked.length ? (
        <p>No eligible selection scores yet.</p>
      ) : (
        <ol className="stage-score-bars">
          {ranked.map((trial) => (
            <li key={trial.id}>
              <span>{trial.id.slice(0, 8)}</span>
              <div className="stage-bar-track">
                <div
                  style={{
                    width: `${Math.max(0, Math.min(1, trial.ranking!.score)) * 100}%`,
                  }}
                />
              </div>
              <strong>{fmt(trial.ranking!.score)}</strong>
            </li>
          ))}
        </ol>
      )}
    </figure>
  );
}

function RetrievalComparison({
  trials,
  rawOnly = false,
}: {
  trials: StageTrial[];
  rawOnly?: boolean;
}) {
  const [metric, setMetric] = useState("ndcg");
  const [k, setK] = useState("10");
  return (
    <div className="stage-figure">
      <h3>
        {rawOnly
          ? "Raw retrieval evidence"
          : "Raw retrieval versus reranked evidence"}
      </h3>
      <div className="actions">
        <label>
          Metric{" "}
          <select
            aria-label="Retrieval comparison metric"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
          >
            {["ndcg", "recall", "precision", "mrr"].map((key) => (
              <option key={key} value={key}>
                {key.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
        <label>
          Top k{" "}
          <select
            aria-label="Retrieval comparison k"
            value={k}
            onChange={(e) => setK(e.target.value)}
          >
            {[3, 5, 10].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
      </div>
      <p>Descriptive means on available questions. Higher is better.</p>
      <DataTable
        name="raw-and-reranked"
        rows={trials.map((trial) => ({
          trial: trial.id.slice(0, 8),
          raw:
            mean(trial, `raw_retrieval_${metric}@${k}`) ??
            trial.metrics[`raw_retrieval_${metric}@${k}`]?.reason ??
            "Not measured",
          ...(!rawOnly
            ? {
                reranked:
                  mean(trial, `reranked_${metric}@${k}`) ??
                  trial.metrics[`reranked_${metric}@${k}`]?.reason ??
                  "Not measured",
              }
            : {}),
        }))}
      />
    </div>
  );
}

function AnswerScatter({ trials }: { trials: StageTrial[] }) {
  const points = trials.filter(
    (t) => t.p95_seconds != null && mean(t, "answer_correctness") != null,
  );
  const maxLatency = Math.max(1, ...points.map((t) => t.p95_seconds!));
  return (
    <figure
      className="stage-figure"
      aria-label="Answer correctness versus p95 latency"
    >
      <figcaption>Answer correctness versus p95 latency</figcaption>
      {!points.length ? (
        <p>Waiting for measured correctness and latency.</p>
      ) : (
        <>
          <svg
            className="stage-scatter"
            viewBox="0 0 540 260"
            role="img"
            aria-label="Correctness versus latency scatter plot; upper left favors quality and speed"
          >
            <line x1="50" y1="220" x2="510" y2="220" stroke="currentColor" />
            <line x1="50" y1="20" x2="50" y2="220" stroke="currentColor" />
            <text x="25" y="28">
              1
            </text>
            <text x="25" y="220">
              0
            </text>
            <text x="50" y="244">
              0 s
            </text>
            <text x="440" y="244">
              {maxLatency.toFixed(1)} s p95
            </text>
            {points.map((trial) => (
              <g key={trial.id}>
                <circle
                  cx={50 + (trial.p95_seconds! / maxLatency) * 430}
                  cy={220 - mean(trial, "answer_correctness")! * 190}
                  r="6"
                  fill="var(--primary)"
                >
                  <title>
                    {trial.id}: correctness{" "}
                    {metricText(trial, "answer_correctness")}, p95{" "}
                    {fmt(trial.p95_seconds)} seconds
                  </title>
                </circle>
                <text
                  x={58 + (trial.p95_seconds! / maxLatency) * 430}
                  y={212 - mean(trial, "answer_correctness")! * 190}
                  fontSize="11"
                >
                  {trial.id.slice(0, 8)}
                </text>
              </g>
            ))}
          </svg>
          <p>
            Upper left favors quality and speed. Exact values are in the trial
            table.
          </p>
        </>
      )}
    </figure>
  );
}

function Confirmation({
  stage,
  sessionId,
}: {
  stage: Stage;
  sessionId: string;
}) {
  const correctness = stage.comparison?.metrics.find(
    (m) => m.metric === "answer_correctness",
  );
  const interval = correctness?.ci95;
  const bound = Math.max(0.01, ...(interval ?? []).map(Math.abs));
  const x = (value: number) => 270 + (value / bound) * 200;
  return (
    <>
      <Notice
        tone={stage.outcome === "confirmed improvement" ? "success" : "info"}
      >
        <strong>{stage.outcome ?? "Confirmation not run"}</strong>.{" "}
        {stage.baseline_retained
          ? "Baseline recommendation retained."
          : "Candidate recommended."}
      </Notice>
      {stage.selected_trial_id && (
        <p>
          Candidate selected from answer trial{" "}
          <TrialLink
            sessionId={sessionId}
            trialId={stage.selected_trial_id}
            phase="answer"
          />
          .
        </p>
      )}
      {stage.comparison?.reason && <p>{stage.comparison.reason}</p>}
      <figure
        className="stage-figure"
        aria-label="Holdout correctness improvement and confidence interval"
      >
        <figcaption>
          Holdout correctness delta: {fmt(correctness?.mean_delta, "Pending")}
        </figcaption>
        {interval && (
          <svg
            className="stage-scatter"
            viewBox="0 0 540 95"
            role="img"
            aria-label={`95% confidence interval ${interval[0].toFixed(3)} to ${interval[1].toFixed(3)}`}
          >
            <line
              x1="270"
              y1="10"
              x2="270"
              y2="65"
              stroke="currentColor"
              strokeDasharray="4 4"
            />
            <line
              x1={x(interval[0])}
              y1="38"
              x2={x(interval[1])}
              y2="38"
              stroke="var(--primary)"
              strokeWidth="5"
            />
            <circle
              cx={x(correctness?.mean_delta ?? 0)}
              cy="38"
              r="7"
              fill="var(--primary)"
            />
            <text x="35" y="86">
              Negative favors baseline
            </text>
            <text x="290" y="86">
              Positive favors candidate
            </text>
          </svg>
        )}
        <p>
          95% interval:{" "}
          {interval?.map((v) => v.toFixed(3)).join(" to ") ?? "Unavailable"}.
          Matched questions: {correctness?.paired_count ?? 0}; missing:{" "}
          {correctness?.missing_count ?? stage.question_count}.
        </p>
      </figure>
      {!!stage.comparison?.metrics.length && (
        <DataTable
          name="holdout-comparison"
          rows={stage.comparison.metrics.map((m) => ({
            metric: m.metric,
            baseline: m.baseline_mean ?? m.reason ?? "Not measured",
            candidate: m.candidate_mean ?? m.reason ?? "Not measured",
            delta: m.mean_delta ?? m.reason ?? "Insufficient evidence",
            ci95: m.ci95?.join(" to ") ?? m.reason ?? "Insufficient evidence",
            paired: m.paired_count,
            missing: m.missing_count,
          }))}
        />
      )}
      {!!stage.configuration_differences?.length && (
        <details>
          <summary>Baseline and candidate configuration differences</summary>
          <DataTable
            name="holdout-configuration-differences"
            rows={stage.configuration_differences.map((change) => ({
              ...change,
            }))}
          />
        </details>
      )}
    </>
  );
}

function TrialTable({
  stage,
  phase,
  sessionId,
  annotateSettings,
  workflowVersion,
}: {
  stage: Stage;
  phase: Phase;
  sessionId: string;
  annotateSettings: boolean;
  workflowVersion: number;
}) {
  const { hash } = useLocation();
  const target = stage.trials.find((t) => hash === `#trial-${t.id}`);
  const ordered = target
    ? [target, ...stage.trials.filter((t) => t.id !== target.id)]
    : stage.trials;
  const rows = ordered.map((trial) => {
    const cfg = trial.configuration ?? {};
    return {
      trial: trial.id,
      status: trial.display_status,
      configuration: describe(cfg),
      selection: trial.roles.join(", ") || "Not selected",
      selection_score: trial.ranking?.score ?? trial.score_reason ?? "Pending",
      descriptive_mean:
        mean(trial, stage.primary_metric) ??
        trial.metrics[stage.primary_metric]?.reason ??
        "Not measured",
      ...(phase !== "retrieval"
        ? {
            claim_support: mean(trial, "claim_support") ?? "Not measured",
            citation_support: mean(trial, "citation_support") ?? "Not measured",
          }
        : {}),
      valid_missing_failed: `${trial.coverage.valid} / ${trial.coverage.missing} / ${trial.coverage.failed}`,
      paired_questions:
        trial.ranking?.paired_count ?? trial.score_reason ?? "Pending",
      p95_seconds:
        trial.p95_seconds ??
        trial.metrics.query_latency_seconds?.reason ??
        "Not measured",
      ...(phase === "retrieval"
        ? {
            chunk_count: mean(trial, "chunk_count") ?? "Not measured",
            chunking: `${describe(cfg.chunker)} (${cfg.chunk_units ?? "units unavailable"})`,
            embedding: describe(cfg.embedding),
            index: describe(cfg.indexers),
            rewriter: describe(cfg.rewriter),
            retriever: describe(cfg.retriever),
          }
        : {
            source_retrieval:
              trial.source_trial_id ?? "Source trial unavailable",
            inherited_chunking: describe(cfg.chunker),
            inherited_retriever: describe(cfg.retriever),
            context: `${describe(cfg.compressor)}; ${describe(cfg.context)}`,
            prompt: describe(
              (cfg.prompt as Config | undefined)?.preset_id ??
                (cfg.prompt as Config | undefined)?.type,
            ),
            context_tokens: mean(trial, "context_tokens") ?? "Not measured",
            prompt_tokens: mean(trial, "prompt_tokens") ?? "Not measured",
            completion_tokens:
              mean(trial, "completion_tokens") ?? "Not measured",
            estimated_cost:
              mean(trial, "total_estimated_cost") ??
              trial.metrics.total_estimated_cost?.reason ??
              "Unknown pricing",
          }),
      reranker: describe(cfg.reranker),
    };
  });
  return (
    <DataTable
      key={`${phase}:${hash}`}
      name={`${phase}-trials`}
      rows={rows}
      cell={(key, row) => {
        const trial = stage.trials.find((t) => t.id === row.trial)!;
        if (key === "trial")
          return (
            <Link id={`trial-${trial.id}`} to={`/results/${trial.id}`}>
              {trial.id.slice(0, 8)}
            </Link>
          );
        if (key === "configuration") {
          const cfg = trial.configuration;
          const chunker = cfg?.chunker as Config | undefined;
          const retriever = cfg?.retriever as Config | undefined;
          const prompt = cfg?.prompt as Config | undefined;
          return (
            <>
              <div className="stage-row-settings">
                {cfg ? (
                  <>
                    <p>
                      <strong>
                        {phase === "retrieval"
                          ? "Chunking"
                          : "Inherited chunking"}
                        :
                      </strong>{" "}
                      {String(chunker?.type ?? "Unknown")} ·{" "}
                      {String(chunker?.chunk_size ?? "Unknown")}{" "}
                      {String(cfg.chunk_units ?? "")} · overlap{" "}
                      {String(chunker?.overlap ?? 0)}
                    </p>
                    <p>
                      <strong>
                        {phase === "retrieval"
                          ? "Retriever"
                          : "Inherited retriever"}
                        :
                      </strong>{" "}
                      {String(retriever?.type ?? "Unknown")}
                    </p>
                    <p>
                      <strong>Reranker:</strong> {describe(cfg.reranker)}
                    </p>
                    {phase !== "retrieval" && (
                      <>
                        <p>
                          <strong>Context:</strong> {describe(cfg.compressor)};{" "}
                          {describe(cfg.context)}
                        </p>
                        <p>
                          <strong>Prompt:</strong>{" "}
                          {String(
                            prompt?.preset_id ?? prompt?.type ?? "Unknown",
                          )}
                        </p>
                      </>
                    )}
                  </>
                ) : (
                  <p>Configuration unavailable</p>
                )}
              </div>
              <Inspector
                sessionId={sessionId}
                trial={trial}
                annotateSettings={annotateSettings}
                workflowVersion={workflowVersion}
              />
            </>
          );
        }
        if (key === "source_retrieval" && trial.source_trial_id)
          return (
            <TrialLink
              sessionId={sessionId}
              trialId={trial.source_trial_id}
              phase="retrieval"
            />
          );
        if (key === "selection")
          return (
            <span title={trial.selection_reasons.join(". ")}>
              {trial.roles.length
                ? trial.roles.map((role) => (
                    <span className="stage-role" key={role}>
                      {role}
                    </span>
                  ))
                : String(row.selection)}
            </span>
          );
        if (typeof row[key] === "number")
          return fmt(
            row[key] as number,
            "Not measured",
            key.includes("cost") ? 6 : 3,
          );
        return undefined;
      }}
    />
  );
}

export function OptimizationStages({
  sessionId,
  status,
  currentStage,
}: {
  sessionId: string;
  status: string;
  currentStage: string;
}) {
  const [search, setSearch] = useSearchParams();
  const requested = search.get("stage");
  const phase: Phase = phases.includes(requested as Phase)
    ? (requested as Phase)
    : phases.includes(currentStage as Phase)
      ? (currentStage as Phase)
      : ["export", "done"].includes(currentStage)
        ? "confirm"
        : "retrieval";
  const active = ![
    "completed",
    "failed",
    "cancelled",
    "budget_exhausted",
  ].includes(status);
  const query = useQuery({
    queryKey: ["optimization-stages", sessionId, status],
    queryFn: () => api<StageDashboard>(`/optimizations/${sessionId}/stages`),
    refetchInterval: active ? 3000 : false,
  });
  const stage = query.data?.stages?.[phase];
  const select = (next: Phase) => {
    const params = new URLSearchParams(search);
    params.set("stage", next);
    setSearch(params);
  };
  return (
    <Card
      title="Stage dashboards"
      description="Trace evidence retrieval, answer optimization, and final holdout confirmation."
    >
      <div
        role="tablist"
        aria-label="Optimization stage"
        className="stage-tabs"
      >
        {phases.map((key, index) => (
          <button
            key={key}
            id={`stage-tab-${key}`}
            role="tab"
            aria-selected={phase === key}
            aria-controls={`stage-panel-${key}`}
            tabIndex={phase === key ? 0 : -1}
            onClick={() => select(key)}
            onKeyDown={(event) => {
              let next: Phase | undefined;
              if (event.key === "ArrowRight")
                next = phases[(index + 1) % phases.length];
              if (event.key === "ArrowLeft")
                next = phases[(index + phases.length - 1) % phases.length];
              if (event.key === "Home") next = phases[0];
              if (event.key === "End") next = phases[2];
              if (next) {
                event.preventDefault();
                select(next);
                document.getElementById(`stage-tab-${next}`)?.focus();
              }
            }}
          >
            {index + 1}. {names[key]}
            <small>{query.data?.stages?.[key]?.status ?? "Loading"}</small>
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        id={`stage-panel-${phase}`}
        aria-labelledby={`stage-tab-${phase}`}
      >
        <h3>
          {names[phase]} / {phase === "confirm" ? "holdout" : "tuning"}
        </h3>
        <p>{descriptions[phase]}</p>
        {phase === "answer" && stage?.plan?.[0]?.reason && (
          <p>{stage.plan[0].reason}</p>
        )}
        <p className="stage-setting-labels">
          {query.data?.workflow_version === 1
            ? "Historical workflow: these are the original saved trial settings. Expanded stage search applies to new sessions. Source-trial links are inferred only when matching is unambiguous."
            : phase === "retrieval"
              ? (query.data?.workflow_version ?? 1) >= 5
                ? "Varied here: chunking, embeddings/index, rewriting and retrieval. Reranking is disabled. Fixed: compression, context, prompts and answer model."
                : "Varied here: chunking, embeddings/index, rewriting, retrieval and reranking. Fixed: compression, context, prompts and answer model."
              : phase === "answer"
                ? "Inherited: chunking, embeddings/index, rewriting and retrieval. Varied here: reranker, context/compression and answer prompt. Fixed: answer model and generation settings."
                : "Fixed: the selected complete pipeline and baseline. No settings are tuned using holdout results."}
        </p>
        {query.isPending && <Loading label="Loading stage results..." />}
        <ErrorNotice error={query.error} retry={() => void query.refetch()} />
        {stage && (
          <>
            {phase !== "confirm" && stage.planner && (
              <OptimizationHistory
                key={`${sessionId}-${phase}`}
                sessionId={sessionId}
                phase={phase}
                active={active}
                planner={stage.planner}
              />
            )}
            <div className="stage-summary">
              <div>
                <span>Trials</span>
                <strong>{stage.trials.length}</strong>
              </div>
              <div>
                <span>Completed</span>
                <strong>
                  {stage.trials.filter((t) => t.status === "completed").length}
                </strong>
              </div>
              <div>
                <span>{stage.split} questions</span>
                <strong>{stage.question_count}</strong>
              </div>
              <div>
                <span>Question errors across trials</span>
                <strong>
                  {stage.trials.reduce((n, t) => n + t.coverage.failed, 0)}
                </strong>
              </div>
            </div>
            {phase === "confirm" && (
              <Confirmation stage={stage} sessionId={sessionId} />
            )}
            {!stage.trials.length ? (
              <Empty title="No trials in this stage yet">
                Trials begin after validation and earlier stages finish.
              </Empty>
            ) : (
              <>
                {phase !== "confirm" && (
                  <div className="stage-charts">
                    <ScoreBars
                      trials={stage.trials}
                      label={
                        phase === "retrieval"
                          ? stage.primary_metric === "raw_retrieval_ndcg@10"
                            ? "Selection score: raw retrieval NDCG@10"
                            : "Selection score: reranked NDCG@10"
                          : "Selection score: answer correctness"
                      }
                    />
                    {phase === "answer" && (
                      <AnswerScatter trials={stage.trials} />
                    )}
                  </div>
                )}
                {phase === "retrieval" && (
                  <RetrievalComparison
                    trials={stage.trials}
                    rawOnly={(query.data?.workflow_version ?? 1) >= 5}
                  />
                )}
                <h3>{names[phase]} trials</h3>
                <p>
                  Selection scores use shared questions; descriptive means use
                  each trial’s available scores. Valid / missing / failed counts
                  refer to the primary metric. Expand View config for exact
                  settings, all metrics, and selection reasons.
                </p>
                <TrialTable
                  stage={stage}
                  phase={phase}
                  sessionId={sessionId}
                  annotateSettings={query.data?.workflow_version !== 1}
                  workflowVersion={query.data?.workflow_version ?? 1}
                />
              </>
            )}
            {phase === "answer" && stage.plan && stage.plan.length > 0 && (
              <details>
                <summary>
                  Search coverage:{" "}
                  {stage.plan.filter((p) => p.scheduled).length}{" "}
                  {query.data?.workflow_version === 4 ? "selected" : "planned"},{" "}
                  {stage.plan.filter((p) => p.trial_id).length} dispatched,{" "}
                  {stage.plan.filter((p) => !p.scheduled).length}{" "}
                  {query.data?.workflow_version === 4
                    ? "untested"
                    : "outside trial cap"}
                </summary>
                <p>
                  The winner is the best observed configuration. The explicit
                  trial cap may leave prompt, context, or reranker variants
                  untested.
                </p>
                <DataTable
                  name="answer-search-coverage"
                  rows={stage.plan.map((p) => ({
                    variant: p.variant,
                    source_retrieval: p.source_trial_id ?? "Baseline",
                    scheduled: p.scheduled,
                    status: p.status,
                    trial: p.trial_id ?? "Not run",
                  }))}
                />
              </details>
            )}
            {phase === "answer" && !!stage.compilations?.length && (
              <>
                <h3>DSPy prompt compilation</h3>
                <p>
                  Compilation usage is offline optimization work, separate from
                  benchmark answer trials. Ranking uses the shared
                  prompt-validation partition.
                </p>
                <DataTable
                  name="prompt-compilations"
                  rows={stage.compilations.map((job) => ({
                    job: job.id,
                    status: job.status,
                    reserved_tokens: job.usage.tokens,
                    cost: job.usage.cost ?? "Unknown pricing",
                  }))}
                />
                {stage.compilations.map((job) => (
                  <PromptCandidateHistory
                    key={job.id}
                    jobId={job.id}
                    status={job.status}
                    history={job.history}
                  />
                ))}
              </>
            )}
          </>
        )}
      </div>
    </Card>
  );
}
