import { useState } from "react";
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

export function RunConfigurationCard({ sessionId }: { sessionId: string }) {
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
    <Card
      title="Run configuration"
      description="Settings captured when this run started. Configuration views are read-only."
    >
      {query.isPending && <p role="status">Loading run configuration...</p>}
      <ErrorNotice error={query.error} retry={() => void query.refetch()} />
      <details>
        <summary>View run configuration</summary>
        {query.isSuccess &&
          (!query.data?.request || !query.data?.template ? (
            <p>Run configuration is unavailable.</p>
          ) : (
            <>
              <Fields
                entries={[
                  ["Benchmark template", request.config_path],
                  ["DSPy prompt configurations", request.prompt_optimization],
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
                  ["Fixed answer model", template.llm],
                  ...["planner", "synthesizer", "verifier", "judge"].map(
                    (role): [string, unknown] => [
                      `${role[0].toUpperCase() + role.slice(1)} model`,
                      request[role],
                    ],
                  ),
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
                    "Hosted providers allowed",
                    request.allow_hosted ? "Yes" : "No",
                  ],
                  [
                    "Component restrictions",
                    Object.keys(object(request.allowed_components)).length
                      ? request.allowed_components
                      : "Default search space",
                  ],
                ]}
              />
              <YamlDetails label="Saved request YAML" value={request} />
              <YamlDetails label="Saved template YAML" value={template} />
            </>
          ))}
      </details>
    </Card>
  );
}

export function TrialConfiguration({
  sessionId,
  trialId,
  stage,
}: {
  sessionId: string;
  trialId: string;
  stage?: "retrieval" | "answer" | "confirm";
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
                  label(
                    "Reranker",
                    stage === "retrieval" ? "retrieval" : "answer",
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
