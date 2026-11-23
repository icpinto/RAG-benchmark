import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, json } from "../lib/api";
import { Button } from "./ui/button";
import { ErrorNotice, Notice } from "./common";

export interface PromptRecipe {
  preset?: string;
  optimizer?: string;
  seed?: number;
  parameters?: Record<string, unknown>;
  limits?: {
    max_tokens?: number;
    max_seconds?: number;
    max_calls?: number;
    max_cost?: number | null;
  };
  proposal_model?: Record<string, unknown> | null;
}
export interface PromptSettings {
  configs: PromptRecipe[];
  answer_shortlist_size?: number;
}
const presetIds = [
  "bootstrap_small_v1",
  "mipro_zero_shot_v1",
  "mipro_few_shot_v1",
  "gepa_light_v1",
];
const defaultRecipe = (): PromptRecipe => ({
  preset: "mipro_zero_shot_v1",
  seed: 42,
  limits: { max_tokens: 2_000_000, max_seconds: 900, max_calls: 500 },
});

export function PromptOptimizerSettings({
  value,
  onChange,
  onValidityChange,
}: {
  value: PromptSettings;
  onChange: (settings: PromptSettings) => void;
  onValidityChange?: (valid: boolean) => void;
}) {
  const [text, setText] = useState(JSON.stringify(value, null, 2));
  const [error, setError] = useState("");
  useEffect(() => {
    setText(JSON.stringify(value, null, 2));
  }, [value]);
  const enabled = value.configs.length > 0;
  const change = (next: PromptSettings) => {
    setError("");
    onValidityChange?.(true);
    onChange(next);
  };
  return (
    <fieldset>
      <legend>DSPy answer prompt optimization</legend>
      <label className="check-row">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) =>
            change({
              ...value,
              configs: e.target.checked ? [defaultRecipe()] : [],
            })
          }
        />
        Enable DSPy compilation
      </label>
      {enabled && (
        <>
          <p>
            Choose reusable presets. Auto Optimize refines a shortlist within
            answering, running each recipe independently from each pipeline's
            original prompt. Compilation and evaluation reserve part of the
            session budget. Manual compilation uses the selected pipeline.
          </p>
          <label>
            Auto Optimize answer shortlist
            <select
              value={value.answer_shortlist_size ?? 2}
              onChange={(e) =>
                change({
                  ...value,
                  answer_shortlist_size: Number(e.target.value),
                })
              }
            >
              <option value={1}>1 pipeline</option>
              <option value={2}>2 pipelines</option>
              <option value={3}>3 pipelines</option>
            </select>
          </label>
          <p>
            Select distinct feasible context and reranker configurations. The
            shortlist shrinks if the protected session budget cannot fit it.
            Answer trials limit screening; recipe evaluations use reserved
            capacity.
          </p>
          {presetIds.map((id) => (
            <label className="check-row" key={id}>
              <input
                type="checkbox"
                checked={value.configs.some((c) => c.preset === id)}
                onChange={(e) =>
                  change({
                    ...value,
                    configs: e.target.checked
                      ? [...value.configs, { ...defaultRecipe(), preset: id }]
                      : value.configs.filter((c) => c.preset !== id),
                  })
                }
              />
              {id}
            </label>
          ))}
          <ol>
            {value.configs.map((recipe, index) => (
              <li key={index}>
                {recipe.preset || recipe.optimizer || "Custom configuration"}{" "}
                {index > 0 && (
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => {
                      const configs = [...value.configs];
                      [configs[index - 1], configs[index]] = [
                        configs[index],
                        configs[index - 1],
                      ];
                      change({ ...value, configs });
                    }}
                  >
                    Move earlier
                  </Button>
                )}
              </li>
            ))}
          </ol>
          <details>
            <summary>Optimizer parameters, model profiles, and limits</summary>
            <label>
              Prompt optimizer configurations JSON
              <textarea
                rows={14}
                value={text}
                onChange={(e) => {
                  setText(e.target.value);
                  try {
                    const parsed: unknown = JSON.parse(e.target.value);
                    if (
                      !parsed ||
                      typeof parsed !== "object" ||
                      !("configs" in parsed) ||
                      !Array.isArray(parsed.configs) ||
                      !parsed.configs.every(
                        (item) =>
                          item &&
                          typeof item === "object" &&
                          !Array.isArray(item),
                      )
                    )
                      throw new Error(
                        "Expected an object with a configs array",
                      );
                    change(parsed as PromptSettings);
                  } catch (err) {
                    setError(String(err));
                    onValidityChange?.(false);
                  }
                }}
              />
            </label>
            <p>
              Use proposal_model for a separate proposal/reflection model.
              Credentials must reference an api_key_env environment variable.
            </p>
          </details>
          {error && <Notice tone="error">{error}</Notice>}
        </>
      )}
    </fieldset>
  );
}

interface Artifact {
  id: string;
  job_id: string;
  payload?: {
    instructions: string;
    demonstrations: unknown[];
    provenance: Record<string, unknown>;
  };
  summary: {
    preset?: string;
    validation_score?: number;
    demonstration_count?: number;
  };
}
interface Job {
  id: string;
  status: string;
  error?: string;
  attempts: number;
  progress: {
    phase?: string;
    results?: {
      index: number;
      preset?: string;
      status: string;
      error?: string;
      artifact_id?: string;
    }[];
  };
  splits: { training: string[]; validation: string[]; evaluation: string[] };
  usage: {
    tokens: number;
    cost: number | null;
    reported_actual_tokens: number;
  };
  artifacts: Artifact[];
}
const terminal = (status?: string) =>
  [
    "completed",
    "completed_with_errors",
    "failed",
    "cancelled",
    "budget_exhausted",
  ].includes(status || "");

export function PromptCompilationPanel({
  settings,
  onSettingsChange,
  prepare,
  onApply,
}: {
  settings: PromptSettings;
  onSettingsChange: (value: PromptSettings) => void;
  prepare: () => Promise<Record<string, unknown> & { config_path: string }>;
  onApply: (yaml: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [valid, setValid] = useState(true);
  const [caseId, setCaseId] = useState("");
  const [caseIds, setCaseIds] = useState<string[]>([]);
  const [jobId, setJobId] = useState("");
  const [artifactId, setArtifactId] = useState("");
  const [allowHosted, setAllowHosted] = useState(false);
  const library = useQuery({
    queryKey: ["prompt-artifacts", jobId],
    queryFn: () => api<Artifact[]>("/prompt-artifacts"),
    enabled: open,
  });
  const job = useQuery({
    queryKey: ["prompt-compilation", jobId],
    queryFn: () => api<Job>(`/prompt-optimizations/${jobId}`),
    enabled: !!jobId,
    refetchInterval: (query) =>
      terminal(query.state.data?.status) ? false : 2000,
  });
  const artifact = useQuery({
    queryKey: ["prompt-artifact", artifactId],
    queryFn: () => api<Artifact>(`/prompt-artifacts/${artifactId}`),
    enabled: !!artifactId,
  });
  const compile = useMutation({
    mutationFn: async () => {
      const inputs = await prepare();
      const cases = await api<{ id: string }[]>(
        `/prompt-optimization/cases?config_path=${encodeURIComponent(inputs.config_path)}`,
      );
      setCaseIds(cases.map((c) => c.id));
      const selected = cases.length === 1 ? cases[0].id : caseId;
      if (!cases.some((c) => c.id === selected))
        throw new Error(
          "Select a concrete pipeline case below, then compile again.",
        );
      return api<{ id: string }>(
        "/prompt-optimizations",
        json("POST", {
          ...inputs,
          case_id: selected,
          prompt_optimization: settings,
          allow_hosted: allowHosted,
        }),
      );
    },
    onSuccess: (result) => setJobId(result.id),
  });
  const cancel = useMutation({
    mutationFn: () =>
      api(`/prompt-optimizations/${jobId}/cancel`, json("POST", {})),
    onSuccess: () => void job.refetch(),
  });
  const apply = useMutation({
    mutationFn: () =>
      api<{ yaml: string }>(`/prompt-artifacts/${artifactId}/config`),
    onSuccess: (result) => onApply(result.yaml),
  });
  const saved = [
    ...(job.data?.artifacts || []),
    ...(library.data || []),
  ].filter((a, i, all) => all.findIndex((b) => b.id === a.id) === i);
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Compile and reuse DSPy prompts</summary>
      {open && (
        <>
          <PromptOptimizerSettings
            value={settings}
            onChange={onSettingsChange}
            onValidityChange={setValid}
          />
          <p>
            Compilation saves the draft and freezes selected labeled questions
            into training, validation, and evaluation partitions (approximately
            60/20/20). Source evidence groups stay together. Applying a prompt
            loads its pipeline and untouched evaluation snapshot.
          </p>
          <label className="check-row">
            <input
              type="checkbox"
              checked={allowHosted}
              onChange={(e) => setAllowHosted(e.target.checked)}
            />
            Allow source data to be sent to hosted models
          </label>
          {caseIds.length > 1 && (
            <label>
              Compilation pipeline case
              <select
                value={caseId}
                onChange={(e) => setCaseId(e.target.value)}
              >
                <option value="">Select one pipeline</option>
                {caseIds.map((id, index) => (
                  <option value={id} key={id}>
                    Case {index + 1}: {id}
                  </option>
                ))}
              </select>
            </label>
          )}
          <Button
            type="button"
            disabled={
              !valid ||
              !settings.configs.length ||
              compile.isPending ||
              (!!jobId && !terminal(job.data?.status))
            }
            onClick={() => compile.mutate()}
          >
            Compile selected configurations
          </Button>
          <ErrorNotice
            error={
              compile.error ||
              job.error ||
              cancel.error ||
              artifact.error ||
              apply.error ||
              library.error
            }
          />
          {job.data && (
            <>
              <p role="status">
                Compilation {job.data.status}:{" "}
                {job.data.progress.phase || "queued"}. Attempt{" "}
                {job.data.attempts}.
              </p>
              <p>
                Questions: {job.data.splits.training.length} training /{" "}
                {job.data.splits.validation.length} validation /{" "}
                {job.data.splits.evaluation.length} evaluation.
              </p>
              <p>
                Reserved tokens: {job.data.usage.tokens.toLocaleString()};
                reported tokens:{" "}
                {job.data.usage.reported_actual_tokens.toLocaleString()}; cost:{" "}
                {job.data.usage.cost === null
                  ? "Unknown pricing"
                  : job.data.usage.cost.toFixed(4)}
                .
              </p>
              {job.data.error && <Notice tone="error">{job.data.error}</Notice>}
              {job.data.progress.results?.map((result) => (
                <p key={result.index}>
                  {result.preset}: {result.status}
                  {result.error ? ` — ${result.error}` : ""}
                </p>
              ))}
              {!terminal(job.data.status) && (
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => cancel.mutate()}
                >
                  Cancel compilation
                </Button>
              )}
            </>
          )}
          <label>
            Saved compiled prompt
            <select
              value={artifactId}
              onChange={(e) => setArtifactId(e.target.value)}
            >
              <option value="">Select a prompt from either workflow</option>
              {saved.map((a) => (
                <option value={a.id} key={a.id}>
                  {a.summary.preset || "Compiled prompt"} — {a.id.slice(0, 12)}
                </option>
              ))}
            </select>
          </label>
          <Button
            type="button"
            variant="ghost"
            onClick={() => void library.refetch()}
          >
            Refresh saved prompts
          </Button>
          {artifact.data && (
            <>
              <p>
                Validation correctness:{" "}
                {artifact.data.summary.validation_score ?? "Unavailable"};
                demonstrations:{" "}
                {artifact.data.payload?.demonstrations.length ?? 0}.
              </p>
              <details>
                <summary>
                  Inspect instructions, examples, and model settings
                </summary>
                <pre>{JSON.stringify(artifact.data.payload, null, 2)}</pre>
              </details>
              <Button
                type="button"
                disabled={apply.isPending}
                onClick={() => apply.mutate()}
              >
                Apply prompt and evaluation partition
              </Button>
            </>
          )}
        </>
      )}
    </details>
  );
}
