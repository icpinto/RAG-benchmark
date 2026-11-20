import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, json } from "../lib/api";
import { download } from "../lib/utils";
import { useFiles } from "../hooks/data";
import { useDraft } from "../hooks/draft";
import { Button } from "../components/ui/button";
import { Card, ErrorNotice, Notice, PageHeading } from "../components/common";
import {
  RunConfigurationCard,
  TrialConfiguration,
  TrialConfigDetails,
} from "../components/optimization-config";

interface ModelProfile {
  provider: "ollama" | "chat_completions";
  model: string;
  base_url: string;
  api_key_env?: string;
  input_price_per_million?: number;
  output_price_per_million?: number;
}
interface Session {
  id: string;
  status: string;
  stage: string;
  error?: string;
  budget: { tokens: number; active_seconds: number; cost: number | null };
  progress: {
    stop_reason?: string;
    answer_ranking?: Ranked[];
    retrieval_ranking?: Ranked[];
    evaluation_dataset?: DatasetProvenance;
  };
}
interface Ranked {
  trial_id: string;
  score: number;
  metric: string;
  paired_count: number;
  missing_count: number;
  p95_seconds: number | null;
  feasible: boolean;
}
interface Trial {
  id: string;
  phase: string;
  split: string;
  status: string;
}
interface Evidence {
  doc_id: string;
  quote: string;
  start_char: number;
  end_char: number;
}
interface SyntheticQuestion {
  question_id: string;
  question: string;
  expected_answer: string;
  category: string;
  evidence: Evidence[];
  human_reviewed: boolean;
  automated_validated: boolean;
  split: string;
}
interface Dataset {
  frozen: number;
  version: number;
  payload: SyntheticQuestion[];
  evaluation_dataset?: DatasetProvenance;
}
interface DatasetProvenance {
  status: "generated" | "reused";
  dataset_id: string;
  source_dataset_id: string;
  source_session_id: string | null;
  question_count: number;
}
interface Recommendation {
  yaml: string;
  confirmed_improvement: boolean;
  baseline_retained: boolean;
  limitations: string[];
  comparison?: {
    metrics: {
      metric: string;
      mean_delta: number | null;
      ci95: number[] | null;
      paired_count: number;
    }[];
  };
}
const roles = ["planner", "synthesizer", "verifier", "judge"] as const;
type Role = (typeof roles)[number];
const initialModel: ModelProfile = {
  provider: "ollama",
  model: "qwen2.5:32b",
  base_url: "http://100.110.250.66:11434",
};
const terminal = (status?: string) =>
  ["completed", "failed", "cancelled", "budget_exhausted"].includes(
    status || "",
  );

export default function AutoOptimize() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const { setDraft } = useDraft();
  const files = useFiles();
  const configs = useQuery({
    queryKey: ["configs"],
    queryFn: () => api<{ files: string[] }>("/configs"),
  });
  const [configPath, setConfigPath] = useState("default.yaml");
  const [sourceMode, setSourceMode] = useState("all");
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [urls, setUrls] = useState("");
  const [audience, setAudience] = useState("");
  const [task, setTask] = useState("");
  const [examples, setExamples] = useState("");
  const [models, setModels] = useState<Record<Role, ModelProfile>>({
    planner: { ...initialModel },
    synthesizer: { ...initialModel },
    verifier: { ...initialModel },
    judge: { ...initialModel },
  });
  const [allowHosted, setAllowHosted] = useState(false);
  const [review, setReview] = useState(false);
  const [datasetMode, setDatasetMode] = useState<
    "reuse_if_compatible" | "generate_new"
  >("reuse_if_compatible");
  const [questions, setQuestions] = useState(100);
  const [retrievalTrials, setRetrievalTrials] = useState(20);
  const [answerTrials, setAnswerTrials] = useState(5);
  const [minutes, setMinutes] = useState(120);
  const [maxTokens, setMaxTokens] = useState(20_000_000);
  const [maxCost, setMaxCost] = useState("");
  const [maxLatency, setMaxLatency] = useState("");
  const [maxQueryCost, setMaxQueryCost] = useState("");
  const [allowed, setAllowed] = useState("{}");
  const [datasetDraft, setDatasetDraft] = useState("");

  const sessions = useQuery({
    queryKey: ["optimizations"],
    queryFn: () => api<Session[]>("/optimizations"),
    refetchInterval: 5000,
  });
  const session = useQuery({
    queryKey: ["optimization", sessionId],
    queryFn: () => api<Session>(`/optimizations/${sessionId}`),
    enabled: !!sessionId,
    refetchInterval: (query) =>
      terminal(query.state.data?.status) ? false : 3000,
  });
  const active = !!sessionId && !terminal(session.data?.status);
  const dataset = useQuery({
    queryKey: ["optimization-dataset", sessionId],
    queryFn: () => api<Dataset | null>(`/optimizations/${sessionId}/dataset`),
    enabled: !!sessionId,
    refetchInterval: active ? 5000 : false,
  });
  const trials = useQuery({
    queryKey: ["optimization-trials", sessionId, session.data?.status],
    queryFn: () => api<Trial[]>(`/optimizations/${sessionId}/trials`),
    enabled: !!sessionId && !!session.data,
    refetchInterval: active ? 3000 : false,
  });
  const events = useQuery({
    queryKey: ["optimization-events", sessionId],
    queryFn: () =>
      api<{ id: string; kind: string; payload: Record<string, unknown> }[]>(
        `/optimizations/${sessionId}/events`,
      ),
    enabled: !!sessionId,
    refetchInterval: active ? 5000 : false,
  });
  const recommendation = useQuery({
    queryKey: ["optimization-recommendation", sessionId, session.data?.status],
    queryFn: () =>
      api<Recommendation | null>(`/optimizations/${sessionId}/recommendation`),
    enabled: !!sessionId,
  });
  const refresh = () =>
    client.invalidateQueries({
      predicate: (query) =>
        String(query.queryKey[0]).startsWith("optimization") &&
        ![
          "optimization-configuration",
          "optimization-trial-configuration",
        ].includes(String(query.queryKey[0])),
    });
  const start = useMutation({
    mutationFn: () =>
      api<{ id: string }>(
        "/optimizations",
        json("POST", {
          config_path: configPath,
          ...(sourceMode === "selected" ? { source_paths: selectedFiles } : {}),
          urls: urls
            .split("\n")
            .map((s) => s.trim())
            .filter(Boolean),
          audience,
          task,
          example_questions: examples.split("\n").filter(Boolean),
          ...models,
          allow_hosted: allowHosted,
          review_before_freeze: review,
          evaluation_dataset_mode: datasetMode,
          allowed_components: JSON.parse(allowed),
          limits: {
            target_questions: questions,
            retrieval_trials: retrievalTrials,
            answer_trials: answerTrials,
            active_seconds: minutes * 60,
            max_tokens: maxTokens,
            max_cost: maxCost ? Number(maxCost) : null,
            max_query_p95_seconds: maxLatency ? Number(maxLatency) : null,
            max_query_cost: maxQueryCost ? Number(maxQueryCost) : null,
          },
        }),
      ),
    onSuccess: (result) => {
      void refresh();
      navigate(`/auto-optimize/${result.id}`);
    },
  });
  const action = useMutation({
    mutationFn: (name: string) =>
      api<{ id?: string }>(
        `/optimizations/${sessionId}/${name}`,
        json("POST", {}),
      ),
    onSuccess: (result) => {
      void refresh();
      if (result.id) navigate(`/auto-optimize/${result.id}`);
    },
  });
  const saveDataset = useMutation({
    mutationFn: () =>
      api(
        `/optimizations/${sessionId}/dataset`,
        json("PUT", { questions: JSON.parse(datasetDraft) }),
      ),
    onSuccess: () => {
      setDatasetDraft("");
      void refresh();
    },
  });
  const modelChange = (role: Role, change: Partial<ModelProfile>) =>
    setModels((previous) => ({
      ...previous,
      [role]: { ...previous[role], ...change },
    }));
  const ranking = session.data?.progress.answer_ranking?.length
    ? session.data.progress.answer_ranking
    : session.data?.progress.retrieval_ranking || [];
  const canReview =
    ["paused", "needs_input"].includes(session.data?.status || "") &&
    !dataset.data?.frozen;

  return (
    <>
      <PageHeading
        eyebrow="AUTOMATED EXPERIMENTS"
        title="Auto Optimize"
        description="Generate or reuse synthetic evaluation data, compare RAG pipelines, and inspect the evidence behind a recommendation."
      />
      <Notice>
        Generated references are synthetic evaluation data. Automated validation
        and human review are recorded separately. The answer model stays fixed
        during optimization.
      </Notice>
      <ErrorNotice
        error={
          start.error ||
          action.error ||
          session.error ||
          dataset.error ||
          trials.error ||
          recommendation.error ||
          sessions.error ||
          files.error ||
          configs.error
        }
      />
      {!sessionId && (
        <Card
          title="Start an optimization"
          description="Choose a corpus and bounded experiment budget. Optional guidance shapes the questions generated from your data."
        >
          <form
            onSubmit={(event) => {
              event.preventDefault();
              start.mutate();
            }}
            className="optimizer-form"
          >
            <label>
              Benchmark template
              <select
                value={configPath}
                onChange={(e) => setConfigPath(e.target.value)}
              >
                {(configs.data?.files || ["default.yaml"]).map((path) => (
                  <option key={path}>{path}</option>
                ))}
              </select>
            </label>
            <label>
              Source documents
              <select
                value={sourceMode}
                onChange={(e) => setSourceMode(e.target.value)}
              >
                <option value="all">
                  All documents in the template corpus
                </option>
                <option value="selected">Selected library files</option>
              </select>
            </label>
            {sourceMode === "selected" && (
              <fieldset>
                <legend>Select documents</legend>
                {files.data?.files.map((file) => (
                  <label key={file.path} className="check-row">
                    <input
                      type="checkbox"
                      checked={selectedFiles.includes(file.path)}
                      onChange={(e) =>
                        setSelectedFiles(
                          e.target.checked
                            ? [...selectedFiles, file.path]
                            : selectedFiles.filter((p) => p !== file.path),
                        )
                      }
                    />
                    {file.name}
                  </label>
                ))}
              </fieldset>
            )}
            <label>
              Source URLs (one per line; blank uses template URLs)
              <textarea
                value={urls}
                onChange={(e) => setUrls(e.target.value)}
                rows={2}
              />
            </label>
            <label>
              Evaluation questions
              <select
                value={datasetMode}
                onChange={(e) =>
                  setDatasetMode(e.target.value as typeof datasetMode)
                }
              >
                <option value="reuse_if_compatible">
                  Reuse matching questions when available
                </option>
                <option value="generate_new">Generate fresh questions</option>
              </select>
            </label>
            <p>
              Reuse requires matching source contents, audience, task guidance,
              example questions, and target question count. Saved questions and
              tuning / holdout splits stay fixed.
            </p>
            <label>
              Audience (optional)
              <input
                value={audience}
                onChange={(e) => setAudience(e.target.value)}
                placeholder="For example, customer support specialists"
              />
            </label>
            <label>
              Task guidance (optional)
              <textarea
                value={task}
                onChange={(e) => setTask(e.target.value)}
                placeholder="Describe the questions this RAG pipeline should answer"
                rows={2}
              />
            </label>
            <label>
              Example questions (optional, one per line)
              <textarea
                value={examples}
                onChange={(e) => setExamples(e.target.value)}
                rows={2}
              />
            </label>
            <div className="optimizer-grid">
              <label>
                Target validated questions
                <input
                  type="number"
                  min={30}
                  max={1000}
                  value={questions}
                  onChange={(e) => setQuestions(Number(e.target.value))}
                  required
                />
              </label>
              <label>
                Retrieval trial limit
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={retrievalTrials}
                  onChange={(e) => setRetrievalTrials(Number(e.target.value))}
                  required
                />
              </label>
              <label>
                Answer trial limit
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={answerTrials}
                  onChange={(e) => setAnswerTrials(Number(e.target.value))}
                  required
                />
              </label>
              <label>
                Active runtime (minutes)
                <input
                  type="number"
                  min={1}
                  max={1440}
                  value={minutes}
                  onChange={(e) => setMinutes(Number(e.target.value))}
                  required
                />
              </label>
              <label>
                Token cap
                <input
                  type="number"
                  min={1000}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  required
                />
              </label>
              <label>
                Monetary cap (optional)
                <input
                  type="number"
                  min={0.001}
                  step="any"
                  value={maxCost}
                  onChange={(e) => setMaxCost(e.target.value)}
                />
              </label>
              <label>
                Query p95 limit (seconds, optional)
                <input
                  type="number"
                  min={0.001}
                  step="any"
                  value={maxLatency}
                  onChange={(e) => setMaxLatency(e.target.value)}
                />
              </label>
              <label>
                Query cost limit (optional)
                <input
                  type="number"
                  min={0.000001}
                  step="any"
                  value={maxQueryCost}
                  onChange={(e) => setMaxQueryCost(e.target.value)}
                />
              </label>
            </div>
            <label className="check-row">
              <input
                type="checkbox"
                checked={review}
                onChange={(e) => setReview(e.target.checked)}
              />
              Pause for dataset review before experiments
            </label>
            <label className="check-row">
              <input
                type="checkbox"
                checked={allowHosted}
                onChange={(e) => setAllowHosted(e.target.checked)}
              />
              Allow selected source data to be sent to configured hosted
              providers
            </label>
            <details>
              <summary>Model roles and advanced search settings</summary>
              <p>
                Endpoints must be reachable from the worker containers.
                Credentials are read from server environment variables. Set
                explicit prices, including zero for local models, to use a
                monetary cap.
              </p>
              {roles.map((role) => (
                <fieldset key={role}>
                  <legend>{role[0].toUpperCase() + role.slice(1)}</legend>
                  <div className="optimizer-grid">
                    <label>
                      {role} provider
                      <select
                        value={models[role].provider}
                        onChange={(e) =>
                          modelChange(role, {
                            provider: e.target
                              .value as ModelProfile["provider"],
                          })
                        }
                      >
                        <option value="ollama">Ollama</option>
                        <option value="chat_completions">
                          Hosted chat completions
                        </option>
                      </select>
                    </label>
                    <label>
                      {role} model
                      <input
                        value={models[role].model}
                        onChange={(e) =>
                          modelChange(role, { model: e.target.value })
                        }
                        required
                      />
                    </label>
                    <label>
                      {role} endpoint
                      <input
                        value={models[role].base_url}
                        onChange={(e) =>
                          modelChange(role, { base_url: e.target.value })
                        }
                        required
                      />
                    </label>
                    <label>
                      {role} API key environment variable
                      <input
                        value={models[role].api_key_env || ""}
                        onChange={(e) =>
                          modelChange(role, {
                            api_key_env: e.target.value || undefined,
                          })
                        }
                      />
                    </label>
                    <label>
                      {role} input price / million tokens
                      <input
                        type="number"
                        min={0}
                        step="any"
                        value={models[role].input_price_per_million ?? ""}
                        onChange={(e) =>
                          modelChange(role, {
                            input_price_per_million:
                              e.target.value === ""
                                ? undefined
                                : Number(e.target.value),
                          })
                        }
                      />
                    </label>
                    <label>
                      {role} output price / million tokens
                      <input
                        type="number"
                        min={0}
                        step="any"
                        value={models[role].output_price_per_million ?? ""}
                        onChange={(e) =>
                          modelChange(role, {
                            output_price_per_million:
                              e.target.value === ""
                                ? undefined
                                : Number(e.target.value),
                          })
                        }
                      />
                    </label>
                  </div>
                </fieldset>
              ))}
              <label>
                Component allowlists (JSON)
                <textarea
                  value={allowed}
                  onChange={(e) => setAllowed(e.target.value)}
                  rows={3}
                  spellCheck={false}
                />
              </label>
            </details>
            <Button type="submit" disabled={start.isPending}>
              {start.isPending ? "Capturing corpus…" : "Start optimization"}
            </Button>
          </form>
        </Card>
      )}
      {sessionId && session.data && (
        <>
          <Card
            title={`Optimization · ${session.data.status}`}
            description={`Current stage: ${session.data.stage}`}
          >
            <div className="optimizer-grid">
              <p>
                Reserved tokens:{" "}
                <strong>{session.data.budget.tokens.toLocaleString()}</strong>
              </p>
              <p>
                Active time:{" "}
                <strong>
                  {Math.round(session.data.budget.active_seconds / 60)} minutes
                </strong>
              </p>
              <p>
                Cost:{" "}
                <strong>
                  {session.data.budget.cost === null
                    ? "Unknown pricing"
                    : session.data.budget.cost.toFixed(4)}
                </strong>
              </p>
            </div>
            {session.data.error && (
              <Notice tone="warning">{session.data.error}</Notice>
            )}
            {session.data.progress.stop_reason && (
              <Notice tone="warning">
                {session.data.progress.stop_reason}
              </Notice>
            )}
            <div className="actions">
              {!terminal(session.data.status) &&
                !["paused", "needs_input"].includes(session.data.status) && (
                  <Button
                    variant="outline"
                    disabled={action.isPending}
                    onClick={() => action.mutate("pause")}
                  >
                    Pause
                  </Button>
                )}
              {["paused", "needs_input"].includes(session.data.status) && (
                <Button
                  disabled={
                    action.isPending || saveDataset.isPending || !!datasetDraft
                  }
                  onClick={() => action.mutate("resume")}
                >
                  {canReview && dataset.data
                    ? "Approve draft and resume"
                    : "Resume optimization"}
                </Button>
              )}
              {!terminal(session.data.status) && (
                <Button
                  variant="outline"
                  disabled={action.isPending}
                  onClick={() => action.mutate("cancel")}
                >
                  Cancel
                </Button>
              )}
              <Button
                variant="outline"
                disabled={action.isPending}
                onClick={() => action.mutate("fork")}
              >
                Start a new version
              </Button>
              <Button variant="ghost" asChild>
                <Link to="/auto-optimize">New optimization</Link>
              </Button>
            </div>
          </Card>
          <RunConfigurationCard key={sessionId} sessionId={sessionId} />
          <Card title="Currently running configs">
            {trials.isPending && <p role="status">Loading running trials...</p>}
            <ErrorNotice
              error={trials.error}
              retry={() => void trials.refetch()}
            />
            {trials.isSuccess &&
              !trials.data.some((trial) => trial.status === "running") && (
                <p>No trial is currently running.</p>
              )}
            {trials.data
              ?.filter((trial) => trial.status === "running")
              .map((trial) => (
                <article
                  key={`${sessionId}:${trial.id}`}
                  className="optimization-running-config"
                  aria-label={`Running trial ${trial.id}`}
                >
                  <h3>Trial {trial.id}</h3>
                  <p>
                    {trial.phase} / {trial.split} · {trial.status}
                  </p>
                  <TrialConfiguration
                    sessionId={sessionId}
                    trialId={trial.id}
                  />
                </article>
              ))}
          </Card>
          {dataset.data && (
            <Card
              title={`Synthetic evaluation dataset · version ${dataset.data.version}`}
              description={`${dataset.data.payload.length} questions · ${dataset.data.frozen ? "Frozen tuning / holdout splits" : "Draft"}`}
            >
              {dataset.data.evaluation_dataset && (
                <p>
                  {dataset.data.evaluation_dataset.status === "reused"
                    ? "Reused evaluation questions"
                    : "Generated evaluation questions"}
                  {dataset.data.evaluation_dataset.status === "reused" &&
                    dataset.data.evaluation_dataset.source_session_id && (
                      <>
                        {" from "}
                        <Link
                          to={`/auto-optimize/${dataset.data.evaluation_dataset.source_session_id}`}
                        >
                          originating experiment
                        </Link>
                        . Questions and tuning / holdout splits are preserved.
                        Generate fresh questions to create an editable set.
                      </>
                    )}
                </p>
              )}
              <details>
                <summary>Inspect questions and source evidence</summary>
                <div className="optimizer-questions">
                  {dataset.data.payload.map((q, index) => (
                    <article key={q.question_id || index}>
                      <h3>
                        {index + 1}. {q.question}
                      </h3>
                      <p>{q.expected_answer}</p>
                      <small>
                        {q.category} · {q.split} · Automated:{" "}
                        {q.automated_validated ? "validated" : "pending"} ·
                        Human: {q.human_reviewed ? "reviewed" : "unreviewed"}
                      </small>
                      {q.evidence.map((e, i) => (
                        <blockquote key={i}>
                          <strong>{e.doc_id}</strong> ({e.start_char}–
                          {e.end_char})<p>{e.quote}</p>
                        </blockquote>
                      ))}
                    </article>
                  ))}
                </div>
              </details>
              {canReview && (
                <details>
                  <summary>Edit draft questions</summary>
                  <p>
                    Save edits before resuming. Evidence must match the captured
                    sources exactly; edited answers are verified again.
                  </p>
                  <textarea
                    aria-label="Draft evaluation questions JSON"
                    className="optimizer-json"
                    value={
                      datasetDraft ||
                      JSON.stringify(dataset.data.payload, null, 2)
                    }
                    onChange={(e) => setDatasetDraft(e.target.value)}
                    rows={16}
                    spellCheck={false}
                  />
                  <ErrorNotice error={saveDataset.error} />
                  <Button
                    disabled={!datasetDraft || saveDataset.isPending}
                    onClick={() => saveDataset.mutate()}
                  >
                    Save dataset edits
                  </Button>
                </details>
              )}
            </Card>
          )}
          <Card
            title="Trial leaderboard"
            description="Retrieval screening uses NDCG@10; answer trials use semantic correctness on matched tuning questions."
          >
            {!trials.data?.length && (
              <p>Trials begin after the dataset is validated and frozen.</p>
            )}
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Trial</th>
                    <th>Phase</th>
                    <th>Status</th>
                    <th>Score</th>
                    <th>Paired / missing</th>
                    <th>p95 seconds</th>
                    <th>Configuration</th>
                  </tr>
                </thead>
                <tbody>
                  {trials.data?.map((trial) => {
                    const ranked = ranking.find((r) => r.trial_id === trial.id);
                    return (
                      <tr key={trial.id}>
                        <td>
                          <Link to={`/results/${trial.id}`}>
                            {trial.id.slice(0, 8)}
                          </Link>
                        </td>
                        <td>
                          {trial.phase} / {trial.split}
                        </td>
                        <td>
                          {trial.status}
                          {ranked && !ranked.feasible
                            ? " · exceeds limits"
                            : ""}
                        </td>
                        <td>{ranked?.score.toFixed(3) ?? "—"}</td>
                        <td>
                          {ranked
                            ? `${ranked.paired_count} / ${ranked.missing_count}`
                            : "—"}
                        </td>
                        <td>{ranked?.p95_seconds?.toFixed(2) ?? "—"}</td>
                        <td>
                          <TrialConfigDetails
                            key={`${sessionId}:${trial.id}`}
                            sessionId={sessionId}
                            trialId={trial.id}
                          />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
          {recommendation.data && (
            <Card
              title="Recommendation"
              description={
                recommendation.data.confirmed_improvement
                  ? "Candidate improvement confirmed on the held-out questions."
                  : "Baseline retained; candidate improvement has not been confirmed."
              }
            >
              {recommendation.data.limitations.map((text) => (
                <p key={text}>{text}</p>
              ))}
              {recommendation.data.comparison?.metrics
                .filter((m) => m.metric === "answer_correctness")
                .map((m) => (
                  <p key={m.metric}>
                    Holdout correctness delta:{" "}
                    {m.mean_delta?.toFixed(3) ?? "unavailable"}; 95% interval:{" "}
                    {m.ci95?.map((v) => v.toFixed(3)).join(" to ") ??
                      "unavailable"}
                    ; paired questions: {m.paired_count}.
                  </p>
                ))}
              <div className="actions">
                <Button
                  onClick={() => {
                    const text = recommendation.data!.yaml;
                    setDraft({
                      text,
                      saved: "",
                      loaded: true,
                      path: `optimized-${sessionId}.yaml`,
                    });
                    navigate("/benchmark-setup");
                  }}
                >
                  Open in Benchmark Setup
                </Button>
                <Button
                  variant="outline"
                  onClick={() =>
                    download(
                      "recommended.yaml",
                      recommendation.data!.yaml,
                      "application/yaml",
                    )
                  }
                >
                  Download YAML
                </Button>
                <a href={`/api/optimizations/${sessionId}/export/qa`} download>
                  Download QA JSON
                </a>
                <a
                  href={`/api/optimizations/${sessionId}/export/report`}
                  download
                >
                  Download report
                </a>
              </div>
            </Card>
          )}
          <Card title="Activity">
            <details>
              <summary>Inspect tool and stage events</summary>
              <ul>
                {events.data?.slice(-30).map((event) => (
                  <li key={event.id}>
                    {event.kind}: {JSON.stringify(event.payload)}
                  </li>
                ))}
              </ul>
            </details>
            <ErrorNotice error={events.error} />
          </Card>
        </>
      )}
      <Card title="Recent optimizations">
        <ul>
          {sessions.data?.map((item) => (
            <li key={item.id}>
              <Link to={`/auto-optimize/${item.id}`}>
                {item.id.slice(0, 8)}
              </Link>{" "}
              · {item.status} · {item.stage}
            </li>
          ))}
        </ul>
        {!sessions.data?.length && <p>No optimizations yet.</p>}
      </Card>
    </>
  );
}
