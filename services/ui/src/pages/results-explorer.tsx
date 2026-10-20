import { lazy, Suspense, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import { ArrowRight, RefreshCw } from "lucide-react";
import { activeRun, api } from "../lib/api";
import type { RecordValue, Result, Run } from "../lib/types";
import {
  aggregateRows,
  comparisonRows,
  heatmap,
  metricRows,
  numeric,
  retrievalMetrics,
} from "../lib/results";
import { display, isRecord } from "../lib/utils";
import { Button } from "../components/ui/button";
import {
  Card,
  Empty,
  ErrorNotice,
  JsonView,
  Loading,
  Notice,
  PageHeading,
  StatusBadge,
  Tabs,
} from "../components/common";
import { DataTable } from "../components/data-table";
const MetricChart = lazy(() => import("../components/metric-chart"));

function RunGuidance({ run, results }: { run: Run; results: Result[] }) {
  if (run.status === "failed")
    return (
      <Notice tone="error">
        {run.error ||
          "This run failed. Review the run summary and worker logs."}
      </Notice>
    );
  if (run.status === "queued")
    return (
      <Notice>
        This run is queued. If it remains queued, verify the Celery worker is
        consuming the ragbench queue.
      </Notice>
    );
  if (run.status === "running")
    return (
      <Notice>
        This run is running. Status and available results refresh automatically.
      </Notice>
    );
  if (!results.length)
    return (
      <Notice tone="warning">
        No scored results were returned. Inspect the run summary and worker
        logs.
      </Notice>
    );
  return Number(run.summary?.errors) > 0 ? (
    <Notice tone="warning">
      This run completed with {String(run.summary?.errors)} case errors.
      Successful results remain available; inspect the summary for failures.
    </Notice>
  ) : null;
}
export function TraceExplorer({ results }: { results: Result[] }) {
  const [index, setIndex] = useState(0);
  const [tab, setTab] = useState("context");
  const result = results[index] || results[0];
  const trace = result?.case_config?.evaluation_step;
  return (
    <Card
      title="Trace Explorer"
      description="Inspect the evidence and prompt behind an individual answer."
    >
      <label className="field">
        Scored result to inspect
        <select
          value={Math.min(index, results.length - 1)}
          onChange={(event) => setIndex(Number(event.target.value))}
        >
          {results.map((row, index) => (
            <option key={index} value={index}>
              {index + 1}. {row.case_id} · {row.question}
            </option>
          ))}
        </select>
      </label>
      <details className="chunk">
        <summary>Question and generated answer</summary>
        <h3>{result?.question}</h3>
        <pre className="code-panel">{result?.answer}</pre>
      </details>
      {!isRecord(trace) || !Object.keys(trace).length ? (
        <Notice>
          This historical result does not include saved context or prompt
          details. Re-run the benchmark to capture evaluation traces.
        </Notice>
      ) : (
        <>
          <Tabs
            label="Trace views"
            value={tab}
            onChange={setTab}
            items={[
              { value: "context", label: "Final context" },
              { value: "prompt", label: "Prompt / query" },
              { value: "chunks", label: "Retrieved chunks" },
              { value: "config", label: "Case configuration" },
            ]}
          />
          {tab === "context" && (
            <pre className="code-panel">{String(trace.context || "")}</pre>
          )}
          {tab === "prompt" && (
            <div className="stack">
              <h3>Rewritten search queries</h3>
              {Array.isArray(trace.rewritten_queries) &&
                trace.rewritten_queries.map((query, i) => (
                  <pre className="code-panel" key={i}>
                    {String(query)}
                  </pre>
                ))}
              <h3>Prompt construction</h3>
              <pre className="code-panel">
                {String(trace.query_build || "")}
              </pre>
            </div>
          )}
          {tab === "chunks" && (
            <div className="stack">
              {Array.isArray(trace.retrieved_context) &&
              trace.retrieved_context.length ? (
                trace.retrieved_context.filter(isRecord).map((chunk, i) => (
                  <details key={i} className="chunk">
                    <summary>
                      Rank {display(chunk.rank)} · {display(chunk.doc_id)}::
                      {display(chunk.chunk_id)}
                      <span>Score {display(chunk.score)}</span>
                    </summary>
                    <JsonView
                      value={{
                        score: chunk.score,
                        retriever: chunk.retriever,
                        metadata: chunk.metadata,
                      }}
                    />
                    <pre className="code-panel">{String(chunk.text || "")}</pre>
                  </details>
                ))
              ) : (
                <Empty title="No saved chunks" />
              )}
            </div>
          )}
          {tab === "config" && (
            <JsonView value={result.case_config} name="case-config.json" />
          )}
        </>
      )}
    </Card>
  );
}
export function RetrievalHeatmap({ results }: { results: Result[] }) {
  const metrics = retrievalMetrics(results);
  const [selected, setSelected] = useState("");
  const metric = metrics.includes(selected) ? selected : metrics[0];
  const matrix = heatmap(results, metric);
  const values = matrix.cases
    .flatMap((caseId) =>
      matrix.questions.map((question) => matrix.value(caseId, question)),
    )
    .filter((value): value is number => value !== undefined);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  return (
    <Card
      title="Retrieval heatmap"
      description="Compare cases across the same questions. Repeated case/question scores are averaged."
    >
      {!metrics.length ? (
        <Empty title="No numeric retrieval metrics available" />
      ) : (
        <>
          <label className="field">
            Heatmap metric
            <select
              value={metric}
              onChange={(event) => setSelected(event.target.value)}
            >
              {metrics.map((key) => (
                <option key={key}>{key}</option>
              ))}
            </select>
          </label>
          <div className="table-scroll">
            <table className="heatmap">
              <caption>
                {metric} · lighter to darker indicates lower to higher values
              </caption>
              <thead>
                <tr>
                  <th>Case</th>
                  {matrix.questions.map((question, index) => (
                    <th key={question} title={question}>
                      Q{index + 1}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.cases.map((caseId) => (
                  <tr key={caseId}>
                    <th title={caseId}>{caseId}</th>
                    {matrix.questions.map((question) => {
                      const value = matrix.value(caseId, question);
                      const intensity =
                        value === undefined
                          ? 0
                          : maximum === minimum
                            ? 0.5
                            : (value - minimum) / (maximum - minimum);
                      return (
                        <td
                          key={question}
                          title={`${caseId}\n${question}\n${metric}: ${display(value)}`}
                          style={{
                            background:
                              value === undefined
                                ? "#f8fafc"
                                : `rgba(37, 99, 235, ${0.08 + intensity * 0.76})`,
                            color: intensity > 0.55 ? "white" : "#1e293b",
                          }}
                        >
                          {value === undefined ? "—" : value.toFixed(3)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details className="chunk">
            <summary>Question labels</summary>
            <ol>
              {matrix.questions.map((question, index) => (
                <li key={question}>
                  <strong>Q{index + 1}.</strong> {question}
                </li>
              ))}
            </ol>
          </details>
        </>
      )}
    </Card>
  );
}
function RunComparison({ runs }: { runs: Run[] }) {
  const [params, setParams] = useSearchParams();
  const ids = params.getAll("compare");
  const [metric, setMetric] = useState("");
  const comparison = useQuery({
    queryKey: ["comparison", ids],
    enabled: ids.length >= 2,
    queryFn: () =>
      api<{
        runs: Run[];
        results: Record<string, Result[]>;
        deltas: RecordValue[];
      }>(
        `/benchmarks/compare?${new URLSearchParams(ids.map((id) => ["run_ids", id]))}`,
      ),
    refetchInterval: (query) =>
      query.state.data?.runs.some((run) => activeRun(run.status))
        ? 5000
        : false,
  });
  const data = comparison.data;
  const metrics = data
    ? Array.from(
        new Set(
          Object.values(data.results).flatMap((rows) =>
            rows.flatMap((row) =>
              Object.entries(row.metrics || {})
                .filter(([, value]) => typeof value === "number")
                .map(([key]) => key),
            ),
          ),
        ),
      )
    : [];
  const chosen = metrics.includes(metric) ? metric : metrics[0];
  const chart =
    data?.runs.flatMap((run) => {
      const values = (data.results[run.id] || [])
        .map((row) => numeric(row.metrics?.[chosen]))
        .filter((v): v is number => v !== undefined);
      return values.length
        ? [
            {
              name: `${run.name} · ${run.id.slice(0, 8)}`,
              value: values.reduce((a, b) => a + b, 0) / values.length,
            },
          ]
        : [];
    }) || [];
  return (
    <Card
      title="Compare benchmark runs"
      description="The first selected run is the baseline. Deltas align by exact case ID and question."
    >
      <div className="selection-list">
        {runs.map((run) => (
          <label key={run.id} className="check-row">
            <input
              type="checkbox"
              checked={ids.includes(run.id)}
              onChange={(event) => {
                const next = new URLSearchParams(params);
                next.delete("compare");
                (event.target.checked
                  ? [...ids, run.id]
                  : ids.filter((id) => id !== run.id)
                ).forEach((id) => next.append("compare", id));
                setParams(next, { flushSync: true, preventScrollReset: true });
              }}
            />
            <span>
              {run.name} · {run.id} <StatusBadge status={run.status} />
              {ids[0] === run.id && (
                <span className="draft-badge">Baseline</span>
              )}
            </span>
          </label>
        ))}
      </div>
      <ErrorNotice
        error={comparison.error}
        retry={() => {
          void comparison.refetch();
        }}
      />
      {ids.length < 2 ? (
        <Empty title="Select at least two runs" />
      ) : comparison.isPending ? (
        <Loading label="Loading comparison…" />
      ) : (
        data && (
          <div className="stack">
            <DataTable
              name="comparison-metadata"
              rows={data.runs.map((run) => ({
                name: run.name,
                run_id: run.id,
                status: run.status,
                config_hash: run.config_hash,
                dataset_hash: run.dataset_hash,
                models: run.model_identifiers,
                artifact_dir: run.artifact_dir,
              }))}
            />
            <div className="comparison-cards">
              {data.runs.map((run) => (
                <div className="component-card" key={run.id}>
                  <h3>{run.name}</h3>
                  <p className="mono muted">{run.id}</p>
                  <DataTable
                    name={`summary-${run.id}`}
                    rows={aggregateRows(run)}
                  />
                </div>
              ))}
            </div>
            {chosen && (
              <>
                <label className="field">
                  Comparison chart metric
                  <select
                    value={chosen}
                    onChange={(event) => setMetric(event.target.value)}
                  >
                    {metrics.map((key) => (
                      <option key={key}>{key}</option>
                    ))}
                  </select>
                </label>
                <p className="field-help">
                  Mean score across available results in each run. Case-aligned
                  deltas appear in the table below.
                </p>
                <Suspense fallback={<Loading />}>
                  <MetricChart data={chart} />
                </Suspense>
              </>
            )}
            <DataTable
              name="run-comparison"
              rows={comparisonRows(data.runs, data.results)}
            />
          </div>
        )
      )}
    </Card>
  );
}
function Summary({ run }: { run: Run }) {
  const aggregate = aggregateRows(run);
  const means = aggregate
    .filter(
      (row) => row.metric.endsWith(".mean") && numeric(row.value) !== undefined,
    )
    .map((row) => ({
      name: row.metric.slice(0, -5),
      value: numeric(row.value)!,
    }));
  const [metric, setMetric] = useState("");
  const selected = means.find((row) => row.name === metric) || means[0];
  return (
    <div className="stack">
      <div className="stat-grid">
        {["cases", "questions", "errors"].map((key) => (
          <div className="stat-card" key={key}>
            <span>{key}</span>
            <strong>{display(run.summary?.[key])}</strong>
          </div>
        ))}
      </div>
      <Card title="Aggregate metrics">
        {selected && (
          <>
            <label className="field">
              Summary chart metric
              <select
                value={selected.name}
                onChange={(event) => setMetric(event.target.value)}
              >
                {means.map((row) => (
                  <option key={row.name}>{row.name}</option>
                ))}
              </select>
            </label>
            <Suspense fallback={<Loading />}>
              <MetricChart data={[selected]} />
            </Suspense>
          </>
        )}
        {aggregate.length ? (
          <DataTable rows={aggregate} name="aggregate-metrics" />
        ) : (
          <Empty title="Aggregate metrics are not available yet" />
        )}
      </Card>
      <Card title="Run metadata and diagnostics">
        <JsonView value={run} name={`${run.id}-summary.json`} />
      </Card>
    </div>
  );
}
export default function ResultsExplorer() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const view = params.get("view") || "summary";
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: () => api<Run[]>("/benchmarks"),
    refetchInterval: (query) =>
      query.state.data?.some((run) => activeRun(run.status)) ? 3000 : false,
  });
  const run = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api<Run>(`/benchmarks/${encodeURIComponent(runId!)}`),
    enabled: !!runId,
    refetchInterval: (query) =>
      activeRun(query.state.data?.status) ? 3000 : false,
  });
  const results = useQuery({
    queryKey: ["results", runId, run.data?.status],
    queryFn: () =>
      api<Result[]>(`/benchmarks/${encodeURIComponent(runId!)}/results`),
    enabled: !!runId && !!run.data,
    refetchInterval: activeRun(run.data?.status) ? 5000 : false,
  });
  const setView = (view: string) => {
    const next = new URLSearchParams(params);
    next.set("view", view);
    setParams(next, { preventScrollReset: true });
  };
  const runList = runs.data || [];
  const detail = results.data || [];
  return (
    <>
      <PageHeading
        eyebrow="03 / Analyze"
        title="Results Explorer"
        description="Follow experiments from queue to insight. Compare performance and inspect the evidence."
        actions={
          <Button
            variant="outline"
            onClick={() => {
              void client.invalidateQueries({ queryKey: ["runs"] });
              void client.invalidateQueries({ queryKey: ["run"] });
              void client.invalidateQueries({ queryKey: ["results"] });
              void client.invalidateQueries({ queryKey: ["comparison"] });
            }}
          >
            <RefreshCw size={15} />
            Refresh runs
          </Button>
        }
      />
      <Card
        title="Benchmark runs"
        description="Recent experiments, configuration identities, and execution status."
      >
        <ErrorNotice
          error={runs.error}
          retry={() => {
            void runs.refetch();
          }}
        />
        {runs.isPending ? (
          <Loading />
        ) : runList.length ? (
          <DataTable
            name="runs"
            rows={runList.map((run) => ({
              run_id: run.id,
              name: run.name,
              status: run.status,
              created_at: run.created_at,
              config_hash: run.config_hash,
              dataset_hash: run.dataset_hash,
            }))}
            onRowClick={(row) => {
              void navigate(`/results/${row.run_id}`);
            }}
            cell={(key, row) =>
              key === "status" ? (
                <StatusBadge status={String(row.status)} />
              ) : undefined
            }
          />
        ) : (
          <Empty title="Your first experiment awaits">
            Create a configuration and launch a benchmark to see results here.
          </Empty>
        )}
        <div className="actions justify-end">
          <Button asChild variant="ghost">
            <Link to="/benchmark-setup">
              Configure a benchmark
              <ArrowRight size={15} />
            </Link>
          </Button>
        </div>
      </Card>
      <Tabs
        label="Results views"
        value={view}
        onChange={setView}
        items={[
          { value: "summary", label: "Run summary" },
          { value: "scores", label: "Scored results & traces" },
          { value: "heatmap", label: "Retrieval heatmap" },
          { value: "compare", label: "Compare runs" },
          { value: "raw", label: "Raw JSON" },
        ]}
      />
      {view === "compare" ? (
        <RunComparison runs={runList} />
      ) : !runId ? (
        <Empty title="Choose a run to explore">
          Open a run from the table above to inspect its metrics, traces, and
          raw results.
        </Empty>
      ) : (
        <div className="stack">
          <ErrorNotice
            error={run.error || results.error}
            retry={() => {
              void run.refetch();
              void results.refetch();
            }}
          />
          {(run.isPending || (!!run.data && results.isPending)) &&
          !run.error &&
          !results.error ? (
            <Loading label="Loading run details…" />
          ) : (
            run.data && (
              <>
                <div className="selected-run">
                  <div>
                    <span className="eyebrow">Selected experiment</span>
                    <h2>{run.data.name}</h2>
                    <code>{run.data.id}</code>
                  </div>
                  <StatusBadge status={run.data.status} />
                </div>
                <RunGuidance run={run.data} results={detail} />
                {view === "summary" && <Summary run={run.data} />}
                {view === "scores" &&
                  (detail.length ? (
                    <>
                      <Card title="Scored answers">
                        <DataTable
                          rows={metricRows(detail)}
                          name="scored-results"
                        />
                      </Card>
                      <TraceExplorer key={runId} results={detail} />
                    </>
                  ) : (
                    <Empty title="No scored results available yet" />
                  ))}
                {view === "heatmap" && <RetrievalHeatmap results={detail} />}
                {view === "raw" && (
                  <Card title="Raw scored results">
                    <JsonView value={detail} name={`${runId}-results.json`} />
                  </Card>
                )}
              </>
            )
          )}
        </div>
      )}
    </>
  );
}
