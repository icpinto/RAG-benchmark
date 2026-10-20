import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  FileText,
  Files,
  ListChecks,
  Plus,
  UploadCloud,
  Save,
  Replace,
  Pencil,
  Trash2,
  LoaderCircle,
} from "lucide-react";
import { api, json } from "../lib/api";
import type { Question } from "../lib/types";
import { useFiles, useQuestions } from "../hooks/data";
import { Button } from "../components/ui/button";
import { Confirm, Modal } from "../components/ui/dialog";
import {
  Card,
  Empty,
  ErrorNotice,
  Loading,
  Notice,
  PageHeading,
  Tabs,
} from "../components/common";
import { DataTable } from "../components/data-table";

const blank: Question = {
  question: "",
  expected_answer: "",
  ground_truth_doc_ids: [],
  ground_truth_chunk_ids: [],
  expected_citations: [],
};
function QuestionForm({
  initial,
  pending,
  onSubmit,
}: {
  initial: Question;
  pending: boolean;
  onSubmit: (question: Question) => void;
}) {
  const [question, setQuestion] = useState(initial);
  const [ids, setIds] = useState({
    ground_truth_doc_ids: initial.ground_truth_doc_ids.join(", "),
    ground_truth_chunk_ids: initial.ground_truth_chunk_ids.join(", "),
    expected_citations: initial.expected_citations.join(", "),
  });
  function submit(event: FormEvent) {
    event.preventDefault();
    onSubmit({
      ...question,
      ...Object.fromEntries(
        Object.entries(ids).map(([key, value]) => [
          key,
          value
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean),
        ]),
      ),
    });
  }
  return (
    <form className="stack" onSubmit={submit}>
      <label className="field">
        Evaluation question
        <textarea
          required
          rows={3}
          value={question.question}
          onChange={(event) =>
            setQuestion({ ...question, question: event.target.value })
          }
        />
      </label>
      <label className="field">
        Expected reference answer
        <textarea
          rows={4}
          value={question.expected_answer}
          onChange={(event) =>
            setQuestion({ ...question, expected_answer: event.target.value })
          }
        />
      </label>
      {Object.entries(ids).map(([key, value]) => (
        <label className="field" key={key}>
          {key.replaceAll("_", " ")}
          <input
            value={value}
            onChange={(event) => setIds({ ...ids, [key]: event.target.value })}
          />
          <span className="field-help">
            Separate multiple values with commas.
          </span>
        </label>
      ))}
      <div className="actions justify-end">
        <Button type="submit" disabled={pending || !question.question.trim()}>
          {pending ? (
            <LoaderCircle
              aria-hidden="true"
              focusable="false"
              className="spin"
              size={16}
            />
          ) : (
            <Save aria-hidden="true" focusable="false" size={16} />
          )}
          {pending ? "Saving…" : "Save question"}
        </Button>
      </div>
    </form>
  );
}
export default function KnowledgeBase() {
  const client = useQueryClient();
  const files = useFiles();
  const questions = useQuestions();
  const [tab, setTab] = useState("documents");
  const [selected, setSelected] = useState("");
  const [sources, setSources] = useState<File[]>([]);
  const [groundTruth, setGroundTruth] = useState<File | null>(null);
  const [sourceInputKey, setSourceInputKey] = useState(0);
  const [truthInputKey, setTruthInputKey] = useState(0);
  const [replace, setReplace] = useState(false);
  const [editing, setEditing] = useState<number | "new" | null>(null);
  const [deleting, setDeleting] = useState<number | null>(null);
  const [message, setMessage] = useState("");
  const [localError, setLocalError] = useState("");
  const preview = useQuery({
    queryKey: ["preview", selected],
    queryFn: () =>
      api<{ path: string; content: string }>(
        `/data/files/content?path=${encodeURIComponent(selected)}`,
      ),
    enabled: !!selected,
  });
  const upload = useMutation({
    mutationFn: async () => {
      if (!sources.length) throw new Error("Choose source documents first.");
      if (
        sources.some(
          (file) => !/\.(txt|md|markdown|pdf|html)$/i.test(file.name),
        )
      )
        throw new Error("Choose TXT, Markdown, PDF, or HTML files.");
      const body = new FormData();
      sources.forEach((file) => body.append("files", file));
      return api("/data/upload", { method: "POST", body });
    },
    onSuccess: () => {
      setMessage("Source documents uploaded.");
      setSources([]);
      setSourceInputKey((value) => value + 1);
      void client.invalidateQueries({ queryKey: ["files"] });
      void client.invalidateQueries({ queryKey: ["preview"] });
    },
  });
  const replaceQuestions = useMutation({
    mutationFn: async () => {
      if (!groundTruth)
        throw new Error("Choose a ground-truth JSON file first.");
      if (groundTruth.size > 10 * 1024 * 1024)
        throw new Error("Ground-truth JSON must be 10 MiB or smaller.");
      const body = new FormData();
      body.append("file", groundTruth, "qa.json");
      return api("/data/ground-truth", { method: "POST", body });
    },
    onSuccess: () => {
      setMessage("Ground-truth questions replaced.");
      setGroundTruth(null);
      setTruthInputKey((value) => value + 1);
      setEditing(null);
      void client.invalidateQueries({ queryKey: ["questions"] });
    },
  });
  const saveQuestion = useMutation({
    mutationFn: (value: Question) =>
      api(
        editing === "new" ? "/data/questions" : `/data/questions/${editing}`,
        json(editing === "new" ? "POST" : "PUT", value),
      ),
    onSuccess: () => {
      setMessage("Question saved.");
      setEditing(null);
      void client.invalidateQueries({ queryKey: ["questions"] });
    },
  });
  const deleteQuestion = useMutation({
    mutationFn: (index: number) =>
      api(`/data/questions/${index}`, { method: "DELETE" }),
    onSuccess: () => {
      setMessage("Question deleted.");
      setEditing(null);
      void client.invalidateQueries({ queryKey: ["questions"] });
    },
  });
  const list = questions.data?.questions || [];
  return (
    <>
      <PageHeading
        eyebrow="01 / Prepare"
        title="Knowledge Base"
        description="Build your corpus and define what a great answer looks like."
      />
      <div className="library-overview">
        <div className="stat-grid" role="group" aria-label="Library totals">
          <div className="stat-card">
            <span>Source documents</span>
            <strong>{files.data?.files.length ?? "—"}</strong>
            <small>Your benchmark knowledge base</small>
          </div>
          <div className="stat-card">
            <span>Evaluation questions</span>
            <strong>{questions.data?.questions.length ?? "—"}</strong>
            <small>Reference answers and supporting evidence</small>
          </div>
        </div>
        <div className="stat-card accent-card">
          <FileText aria-hidden="true" focusable="false" size={24} />
          <span>Better inputs, clearer results</span>
          <small>
            Use focused documents and questions that test real retrieval
            challenges.
          </small>
        </div>
      </div>
      {message && <Notice tone="success">{message}</Notice>}
      <ErrorNotice
        error={
          localError ||
          upload.error ||
          replaceQuestions.error ||
          deleteQuestion.error
        }
      />
      <Card
        title="Add benchmark inputs"
        description="Upload source documents or replace the evaluation question set."
      >
        <div className="two-columns">
          <div className="upload-panel">
            <UploadCloud aria-hidden="true" focusable="false" size={24} />
            <label className="field">
              Source documents
              <input
                key={sourceInputKey}
                type="file"
                multiple
                accept=".txt,.md,.markdown,.pdf,.html"
                onChange={(event) =>
                  setSources(Array.from(event.target.files || []))
                }
              />
            </label>
            <p>TXT, Markdown, PDF, HTML · multiple files supported</p>
            <Button
              variant="outline"
              disabled={upload.isPending || !sources.length}
              onClick={() => {
                setMessage("");
                upload.mutate();
              }}
            >
              {upload.isPending ? (
                <LoaderCircle
                  aria-hidden="true"
                  focusable="false"
                  className="spin"
                  size={16}
                />
              ) : (
                <UploadCloud aria-hidden="true" focusable="false" size={16} />
              )}
              {upload.isPending ? "Uploading…" : "Upload source documents"}
            </Button>
          </div>
          <div className="upload-panel">
            <UploadCloud aria-hidden="true" focusable="false" size={24} />
            <label className="field">
              Ground-truth JSON
              <input
                key={truthInputKey}
                type="file"
                accept=".json"
                onChange={(event) =>
                  setGroundTruth(event.target.files?.[0] || null)
                }
              />
            </label>
            <p>A UTF-8 JSON array of questions · maximum 10 MiB</p>
            <Button
              variant="outline"
              disabled={replaceQuestions.isPending || !groundTruth}
              onClick={() => {
                setLocalError("");
                setReplace(true);
              }}
            >
              {replaceQuestions.isPending ? (
                <LoaderCircle
                  aria-hidden="true"
                  focusable="false"
                  className="spin"
                  size={16}
                />
              ) : (
                <Replace aria-hidden="true" focusable="false" size={16} />
              )}
              {replaceQuestions.isPending
                ? "Uploading…"
                : "Replace ground-truth JSON"}
            </Button>
          </div>
        </div>
      </Card>
      <Tabs
        label="Knowledge base views"
        value={tab}
        onChange={setTab}
        items={[
          { value: "documents", label: "Source documents", icon: Files },
          {
            value: "questions",
            label: "Evaluation questions",
            icon: ListChecks,
          },
        ]}
      />
      {tab === "documents" ? (
        <Card title="Document library" description={files.data?.data_root}>
          <ErrorNotice
            error={files.error}
            retry={() => {
              void files.refetch();
            }}
          />
          {files.isPending ? (
            <Loading />
          ) : !files.data?.files.length ? (
            <Empty title="Your corpus starts here">
              Upload documents to prepare your first benchmark.
            </Empty>
          ) : (
            <DataTable
              name="documents"
              rows={files.data.files.map((file) => ({
                name: file.name,
                path: file.path,
                size_bytes: file.size_bytes,
                modified_at: new Date(file.modified_at * 1000).toLocaleString(),
              }))}
              onRowClick={(row) => setSelected(String(row.path))}
            />
          )}
          <label className="field">
            Preview source document
            <select
              value={selected}
              onChange={(event) => setSelected(event.target.value)}
            >
              <option value="">Choose a document…</option>
              {files.data?.files.map((file) => (
                <option key={file.path} value={file.path}>
                  {file.path}
                </option>
              ))}
            </select>
          </label>
          {selected && (
            <div className="stack">
              {preview.isPending ? (
                <Loading label="Loading preview…" />
              ) : preview.error ? (
                <Notice tone="warning">
                  {preview.error.message} Text previews support UTF-8 files up
                  to 1 MB. Binary documents can still be used in benchmarks.
                </Notice>
              ) : (
                <pre className="code-panel source-preview">
                  {preview.data?.content}
                </pre>
              )}
            </div>
          )}
        </Card>
      ) : (
        <Card title="Evaluation questions" description={questions.data?.path}>
          <div className="actions justify-end">
            <Button
              onClick={() => {
                saveQuestion.reset();
                setEditing("new");
              }}
            >
              <Plus aria-hidden="true" focusable="false" size={16} />
              Add question
            </Button>
          </div>
          <ErrorNotice
            error={questions.error}
            retry={() => {
              void questions.refetch();
            }}
          />
          {questions.isPending ? (
            <Loading />
          ) : !list.length ? (
            <Empty title="Define your first question">
              Add a question or import an existing ground-truth JSON file.
            </Empty>
          ) : (
            <DataTable
              name="questions"
              wrapColumns={["question", "expected_answer", "description"]}
              rows={list.map((question, index) => ({ index, ...question }))}
              cell={(key, row) =>
                key === "index" ? (
                  <div className="actions">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        saveQuestion.reset();
                        setEditing(Number(row.index));
                      }}
                    >
                      <Pencil aria-hidden="true" focusable="false" size={14} />
                      Edit {Number(row.index) + 1}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setDeleting(Number(row.index))}
                    >
                      <Trash2 aria-hidden="true" focusable="false" size={14} />
                      Delete
                    </Button>
                  </div>
                ) : undefined
              }
            />
          )}
        </Card>
      )}
      <Confirm
        open={replace}
        title="Replace all evaluation questions?"
        description="The uploaded JSON will replace the current qa.json. Existing benchmark results remain available."
        confirmLabel="Replace questions"
        confirmIcon={Replace}
        onCancel={() => setReplace(false)}
        onConfirm={() => {
          setReplace(false);
          replaceQuestions.mutate();
        }}
      />
      <Confirm
        open={deleting !== null}
        title="Delete this evaluation question?"
        description="This removes the question from the current ground-truth dataset."
        confirmLabel="Delete question"
        confirmIcon={Trash2}
        onCancel={() => setDeleting(null)}
        onConfirm={() => {
          if (deleting !== null) deleteQuestion.mutate(deleting);
          setDeleting(null);
        }}
      />
      <Modal
        open={editing !== null}
        title={
          editing === "new"
            ? "Add evaluation question"
            : "Edit evaluation question"
        }
        onOpenChange={(open) => {
          if (!open) setEditing(null);
        }}
      >
        <ErrorNotice error={saveQuestion.error} />
        {editing !== null && (
          <QuestionForm
            key={editing}
            initial={editing === "new" ? blank : { ...blank, ...list[editing] }}
            pending={saveQuestion.isPending}
            onSubmit={(value) => saveQuestion.mutate(value)}
          />
        )}
      </Modal>
    </>
  );
}
