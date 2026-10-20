import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useBlocker, useNavigate } from "react-router-dom";
import {
  Play,
  Save,
  FolderOpen,
  Code2,
  SlidersHorizontal,
  ListChecks,
  ListX,
  Layers,
  CheckCircle2,
  Pencil,
  LoaderCircle,
  LogOut,
} from "lucide-react";
import { api, json } from "../lib/api";
import { parseConfig, updateConfig } from "../lib/config";
import type { Catalog } from "../lib/types";
import { useDraft } from "../hooks/draft";
import { useFiles, useQuestions } from "../hooks/data";
import { Button } from "../components/ui/button";
import { Confirm } from "../components/ui/dialog";
import {
  Card,
  ErrorNotice,
  Loading,
  Notice,
  PageHeading,
  Tabs,
} from "../components/common";
import { Fields, StageEditor } from "../components/config-fields";
const YamlEditor = lazy(() => import("../components/yaml-editor"));
type Mode = "all" | "selected" | "configured";
export function selectionPayload(
  configPath: string,
  sourceMode: Mode,
  sources: string[],
  questionMode: Mode,
  questions: number[],
) {
  return {
    config_path: configPath,
    ...(sourceMode !== "configured" ? { source_paths: sources } : {}),
    ...(questionMode !== "configured" ? { question_indices: questions } : {}),
  };
}
function InputSelection({
  label,
  mode,
  onMode,
  options,
  selected,
  onChange,
}: {
  label: string;
  mode: Mode;
  onMode: (mode: Mode) => void;
  options: { id: string; label: string }[];
  selected: string[];
  onChange: (selected: string[]) => void;
}) {
  return (
    <fieldset className="selection">
      <legend>{label}</legend>
      <select
        aria-label={`${label} mode`}
        value={mode}
        onChange={(event) => onMode(event.target.value as Mode)}
      >
        <option value="all">All library entries</option>
        <option value="selected">Select entries (empty means none)</option>
        <option value="configured">Use inputs from configuration</option>
      </select>
      {mode === "selected" && (
        <div className="selection-list">
          <div className="actions">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onChange(options.map((option) => option.id))}
            >
              <ListChecks aria-hidden="true" focusable="false" size={14} />
              Select all
            </Button>
            <Button variant="ghost" size="sm" onClick={() => onChange([])}>
              <ListX aria-hidden="true" focusable="false" size={14} />
              Clear
            </Button>
          </div>
          {options.map((option) => (
            <label key={option.id} className="check-row">
              <input
                type="checkbox"
                checked={selected.includes(option.id)}
                onChange={(event) =>
                  onChange(
                    event.target.checked
                      ? [...selected, option.id]
                      : selected.filter((id) => id !== option.id),
                  )
                }
              />
              <span>{option.label}</span>
            </label>
          ))}
        </div>
      )}
    </fieldset>
  );
}
export default function BenchmarkSetup() {
  const { draft, setDraft } = useDraft();
  const client = useQueryClient();
  const navigate = useNavigate();
  const [group, setGroup] = useState("experiment");
  const [view, setView] = useState("forms");
  const [path, setPath] = useState(draft.path);
  const [loadPath, setLoadPath] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [sourceMode, setSourceMode] = useState<Mode>("all");
  const [questionMode, setQuestionMode] = useState<Mode>("all");
  const [sources, setSources] = useState<string[]>([]);
  const [questionSelection, setQuestionSelection] = useState<{
    revision: string;
    indices: string[];
  }>({ revision: "", indices: [] });
  const files = useFiles();
  const questions = useQuestions();
  const questionRevision = JSON.stringify(questions.data?.questions || []);
  const selectedQuestions =
    questionSelection.revision === questionRevision
      ? questionSelection.indices
      : [];
  const catalog = useQuery({
    queryKey: ["config-schema"],
    queryFn: () => api<Catalog>("/configs/schema"),
  });
  const configs = useQuery({
    queryKey: ["configs"],
    queryFn: () => api<{ config_root: string; files: string[] }>("/configs"),
  });
  const initial = useQuery({
    queryKey: ["initial-config"],
    queryFn: () =>
      api<{ config_path: string; content: string }>("/configs/content"),
    enabled: !draft.loaded,
  });
  useEffect(() => {
    if (initial.data && !draft.loaded) {
      setDraft({
        text: initial.data.content,
        saved: initial.data.content,
        path: initial.data.config_path,
        loaded: true,
      });
      setPath(initial.data.config_path);
    }
  }, [initial.data, draft.loaded, setDraft]);
  const parsed = useMemo(() => {
    try {
      return parseConfig(draft.text);
    } catch (err) {
      return { ...parseConfig(""), errors: [String(err)], duplicates: [] };
    }
  }, [draft.text]);
  const dirty = draft.text !== draft.saved;
  const launching = useRef(false);
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      !launching.current &&
      dirty &&
      currentLocation.pathname !== nextLocation.pathname,
  );
  const load = useMutation({
    mutationFn: (target: string) =>
      api<{ config_path: string; content: string }>(
        `/configs/content?config_path=${encodeURIComponent(target)}`,
      ),
    onSuccess: (payload) => {
      setDraft({
        text: payload.content,
        saved: payload.content,
        path: payload.config_path,
        loaded: true,
      });
      setPath(payload.config_path);
      setMessage("Configuration loaded.");
      setError("");
    },
  });
  const save = useMutation({
    mutationFn: async () => {
      const content = draft.text;
      return {
        ...(await api<{ config_path: string }>(
          "/configs/content",
          json("PUT", { config_path: path, content }),
        )),
        content,
      };
    },
    onSuccess: (payload) => {
      setDraft((current) => ({
        ...current,
        path: payload.config_path,
        saved: payload.content,
        loaded: true,
      }));
      setPath(payload.config_path);
      setMessage("Configuration saved.");
      void client.invalidateQueries({ queryKey: ["configs"] });
    },
  });
  const run = useMutation({
    mutationFn: async (saveFirst: boolean) => {
      if (
        questionMode === "selected" &&
        questionSelection.revision &&
        questionSelection.revision !== questionRevision
      )
        throw new Error(
          "Questions changed. Review the selected questions or explicitly clear the selection before launching.",
        );
      if (sourceMode !== "configured" && !files.isSuccess)
        throw new Error(
          "Load the document library or choose configuration inputs before launching.",
        );
      if (questionMode !== "configured" && !questions.isSuccess)
        throw new Error(
          "Load evaluation questions or choose configuration inputs before launching.",
        );
      let target = path;
      if (saveFirst) target = (await save.mutateAsync()).config_path;
      return api<{ run_id: string }>(
        "/benchmarks/run",
        json(
          "POST",
          selectionPayload(
            target,
            sourceMode,
            sourceMode === "all"
              ? files.data?.files.map((file) => file.path) || []
              : sources.filter((source) =>
                  files.data?.files.some((file) => file.path === source),
                ),
            questionMode,
            questionMode === "all"
              ? questions.data?.questions.map((_, index) => index) || []
              : selectedQuestions.map(Number),
          ),
        ),
      );
    },
    onSuccess: (payload) => {
      launching.current = true;
      void client.invalidateQueries({ queryKey: ["runs"] });
      void navigate(`/results/${payload.run_id}`);
    },
  });
  const requestLoad = (target: string) => {
    if (dirty) setLoadPath(target);
    else load.mutate(target);
  };
  const change = (next: string) => {
    setDraft((current) => ({ ...current, text: next, loaded: true }));
    setMessage("");
  };
  const pending = load.isPending || save.isPending || run.isPending;
  const selectedStage = catalog.data?.stages.find(
    (stage) => stage.key === group,
  );
  const selectedSection = catalog.data?.sections.find(
    (section) => section.key === group,
  );
  return (
    <>
      <PageHeading
        eyebrow="02 / Configure"
        title="Benchmark Setup"
        description="Shape your pipeline, select your inputs, and launch a reproducible experiment."
        actions={
          <span className={`draft-badge ${dirty ? "dirty" : ""}`}>
            {dirty ? (
              <Pencil aria-hidden="true" focusable="false" size={14} />
            ) : (
              <CheckCircle2 aria-hidden="true" focusable="false" size={14} />
            )}
            {dirty ? "Unsaved changes" : "Saved configuration"}
          </span>
        }
      />
      {message && <Notice tone="success">{message}</Notice>}
      <ErrorNotice
        error={error || load.error || save.error || run.error || initial.error}
      />
      <div className="setup-layout">
        <div className="setup-overview">
          <Card title="Configuration file">
            <ErrorNotice
              error={configs.error}
              retry={() => {
                void configs.refetch();
              }}
            />
            <label className="field">
              Saved configurations
              <select
                value=""
                onChange={(event) => {
                  if (event.target.value) requestLoad(event.target.value);
                }}
                disabled={pending}
              >
                <option value="">Choose saved YAML…</option>
                {configs.data?.files.map((file) => (
                  <option key={file}>{file}</option>
                ))}
              </select>
            </label>
            <label className="field">
              Configuration path
              <input
                value={path}
                onChange={(event) => setPath(event.target.value)}
                placeholder="/app/configs/default.yaml"
              />
            </label>
            <Button
              variant="outline"
              disabled={pending || !path}
              onClick={() => requestLoad(path)}
            >
              <FolderOpen aria-hidden="true" focusable="false" size={16} />
              Load from path
            </Button>
          </Card>
          <Card
            title="Benchmark inputs"
            description="Selections apply to this launch without rewriting your YAML."
          >
            <ErrorNotice
              error={files.error}
              retry={() => {
                void files.refetch();
              }}
            />
            <ErrorNotice
              error={questions.error}
              retry={() => {
                void questions.refetch();
              }}
            />
            <InputSelection
              label="Source documents"
              mode={sourceMode}
              onMode={setSourceMode}
              options={
                files.data?.files.map((file) => ({
                  id: file.path,
                  label: file.path,
                })) || []
              }
              selected={sources}
              onChange={setSources}
            />
            <InputSelection
              label="Evaluation questions"
              mode={questionMode}
              onMode={setQuestionMode}
              options={
                questions.data?.questions.map((question, index) => ({
                  id: String(index),
                  label: `${index + 1}. ${question.question}`,
                })) || []
              }
              selected={selectedQuestions}
              onChange={(indices) =>
                setQuestionSelection({
                  revision: questionRevision,
                  indices,
                })
              }
            />
            {questionMode === "selected" &&
              questionSelection.revision !== "" &&
              questionSelection.revision !== questionRevision && (
                <Notice tone="warning">
                  Questions changed. Review and select the current question
                  indices before launching.
                </Notice>
              )}
          </Card>
        </div>
        <Card className="editor-card">
          <div className="editor-toolbar">
            <Tabs
              label="Configuration editor mode"
              value={view}
              onChange={setView}
              items={[
                {
                  value: "forms",
                  label: "Guided editor",
                  icon: SlidersHorizontal,
                },
                { value: "yaml", label: "YAML source", icon: Code2 },
              ]}
            />
            <div
              className="actions"
              role="group"
              aria-label="Save and launch experiment"
            >
              <Button
                variant="outline"
                disabled={pending || !!parsed.errors.length || !path}
                onClick={() => save.mutate()}
              >
                {save.isPending ? (
                  <LoaderCircle
                    aria-hidden="true"
                    focusable="false"
                    className="spin"
                    size={16}
                  />
                ) : (
                  <Save aria-hidden="true" focusable="false" size={16} />
                )}
                {save.isPending ? "Saving…" : "Save"}
              </Button>
              <Button
                disabled={pending || parsed.errors.length > 0 || !path}
                onClick={() => run.mutate(true)}
              >
                {run.isPending && run.variables === true ? (
                  <LoaderCircle
                    aria-hidden="true"
                    focusable="false"
                    className="spin"
                    size={16}
                  />
                ) : (
                  <Play aria-hidden="true" focusable="false" size={16} />
                )}
                {run.isPending && run.variables === true
                  ? "Launching…"
                  : "Save and run"}
              </Button>
              <Button
                variant="outline"
                disabled={pending || !path}
                onClick={() => run.mutate(false)}
              >
                {run.isPending && run.variables === false ? (
                  <LoaderCircle
                    aria-hidden="true"
                    focusable="false"
                    className="spin"
                    size={16}
                  />
                ) : (
                  <Play aria-hidden="true" focusable="false" size={16} />
                )}
                {run.isPending && run.variables === false
                  ? "Launching…"
                  : "Run saved configuration"}
              </Button>
            </div>
          </div>
          <div className="launch-help">
            <p className="field-help">
              The worker runs the saved configuration. Choose Save and run to
              include your draft.
            </p>
            {dirty && (
              <p className="field-help">
                Run saved configuration ignores your unsaved edits.
              </p>
            )}
          </div>
          {parsed.errors.map((message, index) => (
            <Notice tone="error" key={index}>
              {message} Guided editing is paused until the YAML is valid.
            </Notice>
          ))}
          {!!parsed.duplicates.length && (
            <Notice tone="warning">
              Repeated YAML keys: {parsed.duplicates.join(", ")}. The final
              occurrence is effective, matching the backend. Earlier sections
              are retained.
            </Notice>
          )}
          {initial.isPending && !draft.loaded ? (
            <Loading label="Loading configuration…" />
          ) : view === "yaml" ? (
            <Suspense fallback={<Loading label="Loading YAML editor…" />}>
              <YamlEditor value={draft.text} onChange={change} />
            </Suspense>
          ) : (
            <>
              <ErrorNotice
                error={catalog.error}
                retry={() => {
                  void catalog.refetch();
                }}
              />
              {catalog.isPending ? (
                <Loading label="Loading component catalog…" />
              ) : catalog.data && !parsed.errors.length ? (
                <div className="guided-editor">
                  <div className="field config-section-picker">
                    <label htmlFor="configuration-section">
                      Configuration section
                    </label>
                    <select
                      id="configuration-section"
                      value={group}
                      onChange={(event) => setGroup(event.target.value)}
                    >
                      <optgroup label="Configuration">
                        {catalog.data.sections.map((section) => (
                          <option key={section.key} value={section.key}>
                            {section.label}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="Pipeline components">
                        {catalog.data.stages.map((stage) => (
                          <option key={stage.key} value={stage.key}>
                            {stage.label}
                          </option>
                        ))}
                      </optgroup>
                    </select>
                  </div>
                  <nav
                    className="config-nav"
                    aria-label="Configuration sections"
                  >
                    {catalog.data.sections.map((section) => (
                      <button
                        key={section.key}
                        className={group === section.key ? "active" : ""}
                        aria-current={
                          group === section.key ? "true" : undefined
                        }
                        onClick={() => setGroup(section.key)}
                      >
                        <SlidersHorizontal
                          aria-hidden="true"
                          focusable="false"
                          size={18}
                        />
                        {section.label}
                      </button>
                    ))}
                    <span className="eyebrow">Pipeline components</span>
                    {catalog.data.stages.map((stage) => (
                      <button
                        key={stage.key}
                        className={group === stage.key ? "active" : ""}
                        aria-current={group === stage.key ? "true" : undefined}
                        onClick={() => setGroup(stage.key)}
                      >
                        <Layers
                          aria-hidden="true"
                          focusable="false"
                          size={18}
                        />
                        {stage.label}
                      </button>
                    ))}
                  </nav>
                  <div className="config-content">
                    <div className="card-heading">
                      <div className="eyebrow">Configuration</div>
                      <h2>{selectedSection?.label || selectedStage?.label}</h2>
                    </div>
                    {selectedSection && (
                      <Fields
                        key={group}
                        fields={selectedSection.fields}
                        doc={parsed.doc}
                        path={[selectedSection.key]}
                        onChange={(target, value) => {
                          try {
                            change(updateConfig(draft.text, target, value));
                            setError("");
                          } catch (err) {
                            setError(String(err));
                          }
                        }}
                      />
                    )}
                    {selectedStage && (
                      <StageEditor
                        key={group}
                        stage={selectedStage}
                        doc={parsed.doc}
                        onChange={(target, value) =>
                          change(updateConfig(draft.text, target, value))
                        }
                      />
                    )}
                  </div>
                </div>
              ) : (
                <div className="empty">
                  <Code2 aria-hidden="true" focusable="false" size={28} />
                  <p>Use YAML source to edit this configuration.</p>
                  <Button variant="outline" onClick={() => setView("yaml")}>
                    <Code2 aria-hidden="true" focusable="false" size={16} />
                    Open YAML source
                  </Button>
                </div>
              )}
            </>
          )}
        </Card>
      </div>
      <Confirm
        open={loadPath !== null}
        title="Discard unsaved changes?"
        description="Loading another configuration replaces the current draft."
        confirmLabel="Discard and load"
        confirmIcon={FolderOpen}
        onCancel={() => setLoadPath(null)}
        onConfirm={() => {
          if (loadPath) load.mutate(loadPath);
          setLoadPath(null);
        }}
      />
      <Confirm
        open={blocker.state === "blocked"}
        title="Leave setup with unsaved changes?"
        description="Discard your draft changes and continue to the selected workspace."
        confirmLabel="Discard and leave"
        confirmIcon={LogOut}
        onCancel={() => blocker.reset?.()}
        onConfirm={() => {
          setDraft((current) => ({ ...current, text: current.saved }));
          blocker.proceed?.();
        }}
      />
    </>
  );
}
