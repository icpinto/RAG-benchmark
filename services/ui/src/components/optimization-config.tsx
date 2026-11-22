import { useState, type ReactNode } from "react";
import { ChevronDown, Settings2 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { stringify } from "yaml";
import { api } from "../lib/api";
import { Card, ErrorNotice } from "./common";

type Config = Record<string, unknown>;
interface RunConfiguration {
  request: Config | null;
  template: Config | null;
}

const object = (value: unknown): Config =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Config)
    : {};

function describe(value: unknown): string {
  if (value === undefined || value === null) return "Not configured";
  if (Array.isArray(value))
    return value.length ? value.map(describe).join("; ") : "None";
  if (typeof value === "object") {
    const entries = Object.entries(object(value));
    return entries.length
      ? entries.map(([key, child]) => `${key}: ${describe(child)}`).join(", ")
      : "None";
  }
  return String(value);
}

function Fields({ entries }: { entries: [string, unknown][] }) {
  return (
    <dl className="optimization-config-fields">
      {entries.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{describe(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function YamlDetails({ label, value }: { label: string; value: unknown }) {
  return (
    <details>
      <summary>{label}</summary>
      <pre className="optimization-config-yaml" aria-label={label} tabIndex={0}>
        {stringify(value)}
      </pre>
    </details>
  );
}

export function RunConfigurationCard({
  sessionId,
  datasetContent,
  datasetSummary = "Evaluation dataset pending",
  reviewRequired = false,
}: {
  sessionId: string;
  datasetContent?: ReactNode;
  datasetSummary?: string;
  reviewRequired?: boolean;
}) {
  const [expansion, setExpansion] = useState({
    open: reviewRequired,
    reviewRequired,
  });
  if (expansion.reviewRequired !== reviewRequired) {
    setExpansion({ open: reviewRequired || expansion.open, reviewRequired });
  }
  const query = useQuery({
    queryKey: ["optimization-configuration", sessionId],
    queryFn: () =>
      api<RunConfiguration>(`/optimizations/${sessionId}/configuration`),
    staleTime: Infinity,
    retry: false,
  });
  const request = object(query.data?.request);
  const template = object(query.data?.template);
  const limits = object(request.limits);
  return (
    <Card className="run-configuration-card">
      <details
        open={expansion.open}
        onToggle={(event) => {
          const open = event.currentTarget.open;
          setExpansion((previous) =>
            previous.open === open ? previous : { ...previous, open },
          );
        }}
        className="run-configuration-disclosure"
      >
        <summary className="run-configuration-summary">
          <span className="configuration-icon">
            <Settings2 size={20} aria-hidden="true" />
          </span>
          <div className="configuration-summary-text">
            <h2>Run configuration</h2>
            <span>
              {query.isPending
                ? "Loading configuration…"
                : request.config_path
                  ? String(request.config_path)
                  : "Saved settings"}{" "}
              · {datasetSummary}
            </span>
          </div>
          {reviewRequired && (
            <span className="review-indicator">Review required</span>
          )}
          <ChevronDown
            size={18}
            className="disclosure-chevron"
            aria-hidden="true"
          />
        </summary>
        <div className="run-configuration-content">
          <p className="muted">
            Settings captured when this run started. Configuration views are
            read-only.
          </p>
          {query.isPending && <p role="status">Loading run configuration...</p>}
          <ErrorNotice error={query.error} retry={() => void query.refetch()} />
          {query.isSuccess &&
            (!query.data?.request || !query.data?.template ? (
              <p>Run configuration is unavailable.</p>
            ) : (
              <>
                <section
                  className="configuration-group"
                  aria-label="Sources and guidance"
                >
                  <h3>Sources and guidance</h3>
                  <Fields
                    entries={[
                      ["Benchmark template", request.config_path],
                      [
                        "Source documents",
                        request.source_paths ??
                          "All documents in the template corpus",
                      ],
                      [
                        "Source URLs",
                        Array.isArray(request.urls) && request.urls.length
                          ? request.urls
                          : "Template URLs",
                      ],
                      ["Audience", request.audience],
                      ["Task guidance", request.task],
                      ["Example questions", request.example_questions],
                    ]}
                  />
                </section>
                <section
                  className="configuration-group"
                  aria-label="Models and prompts"
                >
                  <h3>Models and prompts</h3>
                  <Fields
                    entries={[
                      ["Fixed answer model", template.llm],
                      [
                        "DSPy prompt configurations",
                        request.prompt_optimization,
                      ],
                      ...["planner", "synthesizer", "verifier", "judge"].map(
                        (role): [string, unknown] => [
                          `${role[0].toUpperCase() + role.slice(1)} model`,
                          request[role],
                        ],
                      ),
                      [
                        "Hosted providers allowed",
                        request.allow_hosted ? "Yes" : "No",
                      ],
                    ]}
                  />
                </section>
                <section
                  className="configuration-group"
                  aria-label="Budget and constraints"
                >
                  <h3>Budget and constraints</h3>
                  <Fields
                    entries={[
                      ["Target questions", limits.target_questions],
                      [
                        "Evaluation questions",
                        request.evaluation_dataset_mode === "generate_new"
                          ? "Generate fresh questions"
                          : "Reuse matching questions when available",
                      ],
                      ["Retrieval trial limit", limits.retrieval_trials],
                      ["Answer trial limit", limits.answer_trials],
                      [
                        "Active runtime (minutes)",
                        typeof limits.active_seconds === "number"
                          ? limits.active_seconds / 60
                          : undefined,
                      ],
                      ["Token cap", limits.max_tokens],
                      ["Monetary cap", limits.max_cost ?? "No cap"],
                      [
                        "Query p95 limit (seconds)",
                        limits.max_query_p95_seconds ?? "No limit",
                      ],
                      ["Query cost limit", limits.max_query_cost ?? "No limit"],
                      [
                        "Pause for dataset review",
                        request.review_before_freeze ? "Yes" : "No",
                      ],
                      [
                        "Component restrictions",
                        Object.keys(object(request.allowed_components)).length
                          ? request.allowed_components
                          : "Default search space",
                      ],
                    ]}
                  />
                </section>
              </>
            ))}
          {datasetContent && (
            <section
              className="configuration-group evaluation-dataset"
              aria-label="Evaluation dataset"
            >
              <h3>Evaluation dataset</h3>
              {datasetContent}
            </section>
          )}
          {query.data?.request && query.data?.template && (
            <section className="configuration-group" aria-label="Saved YAML">
              <h3>Saved YAML</h3>
              <YamlDetails label="Saved request YAML" value={request} />
              <YamlDetails label="Saved template YAML" value={template} />
            </section>
          )}
        </div>
      </details>
    </Card>
  );
}

export function TrialConfiguration({
  sessionId,
  trialId,
  stage,
  workflowVersion = 1,
}: {
  sessionId: string;
  trialId: string;
  stage?: "retrieval" | "answer" | "confirm";
  workflowVersion?: number;
}) {
  const query = useQuery({
    queryKey: ["optimization-trial-configuration", sessionId, trialId],
    queryFn: () =>
      api<{ config: Config | null }>(
        `/optimizations/${sessionId}/trials/${trialId}/config`,
      ),
    staleTime: Infinity,
    retry: false,
  });
  const config = query.data?.config;
  const pipeline = object(config?.pipeline);
  const ingestion = object(pipeline.ingestion);
  const queryConfig = object(pipeline.query);
  const label = (name: string, group: "retrieval" | "answer") =>
    !stage
      ? name
      : `${name} · ${stage === "confirm" ? "Fixed for confirmation" : group === "retrieval" && stage === "answer" ? "Inherited" : group === "answer" && stage === "retrieval" ? "Fixed" : "Varied here"}`;
  return (
    <div>
      {query.isPending && <p role="status">Loading trial configuration...</p>}
      <ErrorNotice error={query.error} retry={() => void query.refetch()} />
      {query.isSuccess &&
        (config ? (
          <>
            <Fields
              entries={[
                [label("Chunker", "retrieval"), ingestion.chunkers],
                [label("Embedding model", "retrieval"), ingestion.embeddings],
                [label("Index strategy", "retrieval"), ingestion.indexers],
                [label("Retriever", "retrieval"), queryConfig.retrievers],
                [label("Rewriter", "retrieval"), queryConfig.rewriters],
                [
                  stage === "retrieval" && workflowVersion >= 5
                    ? "Reranker · Disabled during retrieval"
                    : label(
                        "Reranker",
                        stage === "retrieval" && workflowVersion < 5
                          ? "retrieval"
                          : "answer",
                      ),
                  queryConfig.rerankers,
                ],
                [label("Compressor", "answer"), queryConfig.compressors],
                [
                  label("Context settings", "answer"),
                  queryConfig.context_assemblers,
                ],
                [
                  label("Prompt settings", "answer"),
                  queryConfig.prompt_builders,
                ],
                ["Fixed answer model and generation settings", config.llm],
                ["Runtime mode", object(config.runtime).mode],
              ]}
            />
            <YamlDetails label="Full trial configuration YAML" value={config} />
          </>
        ) : (
          <p>Trial configuration is unavailable.</p>
        ))}
    </div>
  );
}

export function TrialConfigDetails({
  sessionId,
  trialId,
}: {
  sessionId: string;
  trialId: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>View config</summary>
      {open && <TrialConfiguration sessionId={sessionId} trialId={trialId} />}
    </details>
  );
}
