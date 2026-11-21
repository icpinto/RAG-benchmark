import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { ErrorNotice, Loading } from "./common";

export interface PlannerStatus {
  status: string;
  iteration_count: number;
  stop_reason: string | null;
  error: string | null;
}

interface Iteration {
  id: string;
  updated_at: number;
  sequence: number;
  status: string;
  observation: {
    trial_id: string | null;
    candidate_id: string;
    status: string;
    metric: string;
    score: number | null;
    errors: string[];
  };
  decision: {
    reflection: string;
    action: "run" | "stop";
    hypothesis: string;
    candidate_id?: string | null;
    reason: string;
    memory: {
      lessons: { text: string; status: string; trial_ids: string[] }[];
    };
  } | null;
  attempts: { id: string; status: string; error?: string | null }[];
}

function IterationContext({
  sessionId,
  id,
  version,
}: {
  sessionId: string;
  id: string;
  version: number;
}) {
  const query = useQuery({
    queryKey: ["optimization-iteration", sessionId, id, version],
    queryFn: () =>
      api<Record<string, unknown>>(
        `/optimizations/${sessionId}/iterations/${id}`,
      ),
  });
  return (
    <div>
      {query.isPending && <Loading label="Loading planner context..." />}
      <ErrorNotice error={query.error} retry={() => void query.refetch()} />
      {query.data && <pre>{JSON.stringify(query.data, null, 2)}</pre>}
    </div>
  );
}

export function OptimizationHistory({
  sessionId,
  phase,
  active,
  planner,
}: {
  sessionId: string;
  phase: "retrieval" | "answer";
  active: boolean;
  planner: PlannerStatus;
}) {
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const limit = 10;
  const query = useQuery({
    queryKey: [
      "optimization-iterations",
      sessionId,
      phase,
      offset,
      planner.iteration_count,
      planner.status,
      active,
    ],
    queryFn: () =>
      api<{ items: Iteration[]; total: number }>(
        `/optimizations/${sessionId}/iterations?phase=${phase}&offset=${offset}&limit=${limit}`,
      ),
    refetchInterval: active ? 3000 : false,
  });
  return (
    <section aria-label="Experiment history" className="optimization-history">
      <h3>Experiment history</h3>
      <p>
        Planner: {planner.status}. {planner.iteration_count} observations.
        {planner.stop_reason && ` Stage stopped: ${planner.stop_reason}`}
      </p>
      {planner.error && <p role="status">{planner.error}</p>}
      <p>
        Each observation informs the next experiment. Lessons are the planner’s
        interpretations; measured scores determine ranking.
      </p>
      {query.isPending && <Loading label="Loading experiment history..." />}
      <ErrorNotice error={query.error} retry={() => void query.refetch()} />
      {query.data?.items.length === 0 && <p>No observations recorded yet.</p>}
      {query.data?.items.map((item) => (
        <article key={item.id}>
          <h4>
            Experiment {item.sequence}: {item.observation.status}
          </h4>
          <p>
            {item.observation.metric}:{" "}
            {item.observation.score == null
              ? "Unavailable"
              : item.observation.score.toFixed(3)}
            . Decision: {item.status}.
          </p>
          {item.observation.errors.map((error) => (
            <p key={error}>{error}</p>
          ))}
          {item.decision && (
            <>
              <p>{item.decision.reflection}</p>
              <p>
                <strong>Next action:</strong>{" "}
                {item.decision.action === "stop"
                  ? `Stop — ${item.decision.reason}`
                  : item.decision.hypothesis}
              </p>
              {item.decision.candidate_id && (
                <p>Candidate: {item.decision.candidate_id}</p>
              )}
              {item.decision.memory.lessons.length > 0 && (
                <ul>
                  {item.decision.memory.lessons.map((lesson, index) => (
                    <li key={index}>
                      {lesson.status}: {lesson.text} (trials:{" "}
                      {lesson.trial_ids.join(", ")})
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
          {item.attempts
            .filter((attempt) => attempt.error)
            .map((attempt) => (
              <p key={attempt.id}>Planner attempt failed: {attempt.error}</p>
            ))}
          <button
            type="button"
            aria-expanded={selected === item.id}
            onClick={() => setSelected(selected === item.id ? null : item.id)}
          >
            {selected === item.id
              ? "Hide planner context"
              : "View planner context"}
          </button>
          {selected === item.id && (
            <IterationContext
              sessionId={sessionId}
              id={item.id}
              version={item.updated_at}
            />
          )}
        </article>
      ))}
      {query.data && query.data.total > 0 && (
        <div className="table-toolbar">
          <button
            type="button"
            disabled={offset === 0}
            onClick={() => {
              setOffset(Math.max(0, offset - limit));
              setSelected(null);
            }}
          >
            Previous experiments
          </button>
          <span>
            {offset + 1}–{Math.min(offset + limit, query.data.total)} of{" "}
            {query.data.total}
          </span>
          <button
            type="button"
            disabled={offset + limit >= query.data.total}
            onClick={() => {
              setOffset(offset + limit);
              setSelected(null);
            }}
          >
            Next experiments
          </button>
        </div>
      )}
    </section>
  );
}
