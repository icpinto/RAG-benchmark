import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { ErrorNotice, Loading, Notice } from "./common";
import { Button } from "./ui/button";

interface Evaluation {
  score: number | null;
  sample_count: number | null;
  partition: string;
  scope: string;
  stage: string;
}
interface Candidate {
  id: string;
  config_index: number;
  preset: string | null;
  proposal_order: number;
  kind: string;
  instructions_preview: string;
  demonstration_count: number | null;
  sources: string[];
  evaluations: Evaluation[];
  selected: boolean;
  historical: boolean;
}
interface CandidateDetail extends Candidate {
  instructions: string | null;
  demonstrations:
    { question: string; context: string; answer: string }[] | null;
}
export interface CompilationHistorySummary {
  available: boolean;
  candidate_count: number;
  pending: boolean;
  configurations: {
    index: number;
    preset?: string | null;
    status: string;
    count?: number;
  }[];
}
interface CandidatePage {
  items: Candidate[];
  total: number;
  history: CompilationHistorySummary;
}
const score = (value: number | null) =>
  value == null ? "Unavailable" : value.toFixed(3);
const kindName = (kind: string) =>
  ({
    instruction: "Instruction suggestion",
    demonstrations: "Example set",
    prompt: "Prompt",
  })[kind] ?? kind;

function CandidateContent({
  jobId,
  candidateId,
}: {
  jobId: string;
  candidateId: string;
}) {
  const detail = useQuery({
    queryKey: ["prompt-candidate", jobId, candidateId],
    queryFn: () =>
      api<CandidateDetail>(
        `/prompt-optimizations/${jobId}/candidates/${encodeURIComponent(candidateId)}`,
      ),
  });
  const candidate = detail.data;
  const content = useRef<HTMLElement>(null);
  useEffect(() => {
    if (detail.isSuccess) content.current?.focus();
  }, [detail.isSuccess]);
  return (
    <section
      id={`prompt-candidate-${jobId}`}
      ref={content}
      tabIndex={-1}
      className="prompt-candidate-content"
      aria-label="Prompt candidate details"
    >
      {detail.isPending && <Loading label="Loading full prompt..." />}
      <ErrorNotice error={detail.error} retry={() => void detail.refetch()} />
      {candidate && (
        <>
          <h4>
            Configuration {candidate.config_index + 1}:{" "}
            {candidate.preset || "Custom optimizer"} / suggestion{" "}
            {candidate.proposal_order}
          </h4>
          {candidate.historical && (
            <Notice>
              Candidate history was not recorded. This is the saved selected
              prompt.
            </Notice>
          )}
          {candidate.selected && (
            <p>
              <strong>Selected by DSPy</strong>. Answer-trial and holdout
              selection are separate.
            </p>
          )}
          <h4>Instructions</h4>
          {candidate.instructions != null ? (
            <pre>{candidate.instructions}</pre>
          ) : (
            <p>
              This example set is combined with instructions during prompt
              search.
            </p>
          )}
          {candidate.demonstrations == null ? (
            <p>
              Examples are selected separately. Inspect evaluated prompts to see
              the combinations used.
            </p>
          ) : (
            <details>
              <summary>Examples ({candidate.demonstrations.length})</summary>
              {!candidate.demonstrations.length && (
                <p>No examples in this prompt.</p>
              )}
              {candidate.demonstrations.map((example, index) => (
                <section key={index}>
                  <h4>Example {index + 1}</h4>
                  <strong>Question</strong>
                  <pre>{example.question}</pre>
                  <strong>Context</strong>
                  <pre>{example.context}</pre>
                  <strong>Answer</strong>
                  <pre>{example.answer}</pre>
                </section>
              ))}
            </details>
          )}
          <h4>Evaluations</h4>
          {!candidate.evaluations.length && (
            <p>Not evaluated as a standalone prompt.</p>
          )}
          {candidate.evaluations.map((evaluation, index) => (
            <p key={index}>
              {evaluation.stage === "final_validation"
                ? "Final artifact validation"
                : "Optimizer evaluation"}
              : {score(evaluation.score)}
              {" · "}
              {evaluation.partition},{" "}
              {evaluation.scope === "full" ? "full set" : "minibatch"}
              {" · "}
              {evaluation.sample_count ?? "Unknown number of"} questions
            </p>
          ))}
        </>
      )}
    </section>
  );
}

export function PromptCandidateHistory({
  jobId,
  status,
  history,
}: {
  jobId: string;
  status: string;
  history?: CompilationHistorySummary;
}) {
  const [open, setOpen] = useState(false);
  const [config, setConfig] = useState("");
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState("");
  const limit = 25;
  const query = useQuery({
    queryKey: [
      "prompt-candidates",
      jobId,
      status,
      history?.candidate_count,
      config,
      search,
      offset,
    ],
    queryFn: () => {
      const params = new URLSearchParams({
        offset: String(offset),
        limit: String(limit),
        q: search,
      });
      if (config !== "") params.set("config_index", config);
      return api<CandidatePage>(
        `/prompt-optimizations/${jobId}/candidates?${params}`,
      );
    },
    enabled: open,
  });
  const summary = query.data?.history ?? history;
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        View prompt suggestions — {jobId.slice(0, 12)}
        {history?.available ? ` (${history.candidate_count})` : ""}
      </summary>
      {open && (
        <div className="prompt-candidate-history">
          <p>
            Suggestions appear after each optimizer configuration completes.
            Scores use a 0–1 scale; training and minibatch results have
            different evaluation coverage.
          </p>
          {summary?.pending && (
            <Notice>
              Compilation is in progress. Suggestions from completed
              configurations appear here.
            </Notice>
          )}
          {summary?.configurations.some(
            (item) => item.preset === "bootstrap_small_v1",
          ) && (
            <p>
              BootstrapFewShot selects examples rather than proposing new
              instructions.
            </p>
          )}
          <div className="table-toolbar">
            <label>
              Optimizer configuration
              <select
                value={config}
                onChange={(event) => {
                  setConfig(event.target.value);
                  setOffset(0);
                }}
              >
                <option value="">All configurations</option>
                {summary?.configurations.map((item) => (
                  <option key={item.index} value={item.index}>
                    {item.index + 1}. {item.preset || "Custom optimizer"} (
                    {item.status})
                  </option>
                ))}
              </select>
            </label>
            <label>
              Search prompt suggestions
              <input
                value={search}
                maxLength={500}
                onChange={(event) => {
                  setSearch(event.target.value);
                  setOffset(0);
                }}
              />
            </label>
          </div>
          {query.isPending && <Loading label="Loading prompt suggestions..." />}
          <ErrorNotice error={query.error} retry={() => void query.refetch()} />
          {query.data && (
            <>
              {!query.data.items.length && (
                <p>
                  {summary?.pending
                    ? "No completed suggestions match this search yet."
                    : "No prompt suggestions match this search. History is unavailable for configurations that did not complete successfully."}
                </p>
              )}
              {!!query.data.items.length && (
                <div className="table-scroll">
                  <table aria-label="DSPy prompt suggestions">
                    <thead>
                      <tr>
                        <th>Inspect</th>
                        <th>Configuration</th>
                        <th>Suggestion</th>
                        <th>Instructions</th>
                        <th>Examples</th>
                        <th>Optimizer score</th>
                        <th>Selection</th>
                      </tr>
                    </thead>
                    <tbody>
                      {query.data.items.map((candidate) => {
                        const evaluations = candidate.evaluations.filter(
                          (item) => item.stage === "optimizer",
                        );
                        const evaluation = evaluations.at(-1);
                        return (
                          <tr key={candidate.id}>
                            <td>
                              <Button
                                type="button"
                                variant="ghost"
                                aria-label={`View prompt ${candidate.proposal_order} from configuration ${candidate.config_index + 1}`}
                                aria-controls={`prompt-candidate-${jobId}`}
                                aria-expanded={selected === candidate.id}
                                onClick={() => setSelected(candidate.id)}
                              >
                                View prompt
                              </Button>
                            </td>
                            <td>
                              {candidate.config_index + 1}.{" "}
                              {candidate.preset || "Custom optimizer"}
                            </td>
                            <td>
                              {candidate.proposal_order}.{" "}
                              {kindName(candidate.kind)}
                            </td>
                            <td className="prompt-candidate-preview">
                              {candidate.instructions_preview || "Example set"}
                            </td>
                            <td>
                              {candidate.demonstration_count ??
                                "Chosen separately"}
                            </td>
                            <td>
                              {evaluation ? (
                                <>
                                  {score(evaluation.score)}
                                  <br />
                                  {evaluation.partition} /{" "}
                                  {evaluation.scope === "full"
                                    ? "full set"
                                    : "minibatch"}
                                </>
                              ) : candidate.evaluations.length ? (
                                "See final validation"
                              ) : (
                                "Not evaluated"
                              )}
                            </td>
                            <td>
                              {candidate.selected
                                ? "Selected by DSPy"
                                : "Not selected"}
                              {candidate.historical && (
                                <p>Candidate history was not recorded.</p>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
              <div className="actions">
                <Button
                  type="button"
                  variant="outline"
                  disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - limit))}
                >
                  Previous suggestions
                </Button>
                <span>
                  {query.data.total
                    ? `${offset + 1}–${Math.min(offset + limit, query.data.total)}`
                    : "0"}{" "}
                  of {query.data.total}
                </span>
                <Button
                  type="button"
                  variant="outline"
                  disabled={offset + limit >= query.data.total}
                  onClick={() => setOffset(offset + limit)}
                >
                  Next suggestions
                </Button>
              </div>
            </>
          )}
          {selected && (
            <CandidateContent
              key={selected}
              jobId={jobId}
              candidateId={selected}
            />
          )}
        </div>
      )}
    </details>
  );
}
