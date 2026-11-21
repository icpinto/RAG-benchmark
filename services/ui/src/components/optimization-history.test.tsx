import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, expect, it, vi } from "vitest";
import { OptimizationHistory } from "./optimization-history";

const fetchMock = vi.fn();
const planner = {
  status: "paused",
  iteration_count: 12,
  stop_reason: null,
  error: "Planner failed twice; resume to retry this decision",
};
const item = {
  id: "iteration-1",
  sequence: 1,
  status: "committed",
  updated_at: 1,
  observation: {
    trial_id: "trial-1",
    candidate_id: "candidate-1",
    status: "completed",
    metric: "reranked_ndcg@10",
    score: 0,
    errors: [],
  },
  decision: {
    action: "run",
    reflection: "Relevant evidence is missing",
    hypothesis: "BM25 will improve retrieval",
    candidate_id: null,
    reason: "",
    memory: {
      lessons: [
        {
          text: "Test lexical retrieval",
          status: "tentative",
          trial_ids: ["trial-1"],
        },
      ],
    },
  },
  attempts: [
    {
      id: "attempt-1",
      status: "failed",
      error: "Configuration has already been attempted",
    },
  ],
};

function mount() {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <OptimizationHistory
        sessionId="session"
        phase="retrieval"
        active={false}
        planner={planner}
      />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockReset();
});

it("shows observations, hypotheses, lessons, failures and full saved context", async () => {
  fetchMock.mockImplementation((url: string) =>
    Promise.resolve(
      new Response(
        JSON.stringify(
          url.endsWith("/iteration-1")
            ? { prompt_version: "ragbench-react-v1", prompt: "Saved input" }
            : { items: [item], total: 12 },
        ),
      ),
    ),
  );
  mount();
  expect(
    await screen.findByText("Relevant evidence is missing"),
  ).toBeInTheDocument();
  expect(screen.getByText(/reranked_ndcg@10: 0.000/)).toBeInTheDocument();
  expect(screen.getByText(/BM25 will improve retrieval/)).toBeInTheDocument();
  expect(
    screen.getByText(/tentative: Test lexical retrieval/),
  ).toBeInTheDocument();
  expect(
    screen.getByText(/Configuration has already been attempted/),
  ).toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent("Planner failed twice");
  fireEvent.click(screen.getByRole("button", { name: "View planner context" }));
  expect(await screen.findByText(/Saved input/)).toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/optimizations/session/iterations/iteration-1",
    expect.anything(),
  );
  fireEvent.click(screen.getByRole("button", { name: "Next experiments" }));
  await waitFor(() =>
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/optimizations/session/iterations?phase=retrieval&offset=10&limit=10",
      expect.anything(),
    ),
  );
});

it("shows a stopping reason and leaves missing scores explicit", async () => {
  fetchMock.mockResolvedValue(
    new Response(
      JSON.stringify({
        items: [
          {
            ...item,
            observation: {
              ...item.observation,
              status: "failed",
              score: null,
              errors: ["Trial failed; inspect worker logs"],
            },
            decision: {
              ...item.decision,
              action: "stop",
              reason: "No promising candidates remain",
            },
          },
        ],
        total: 1,
      }),
    ),
  );
  mount();
  expect(
    await screen.findByText(/Stop — No promising candidates remain/),
  ).toBeInTheDocument();
  expect(screen.getByText(/reranked_ndcg@10: Unavailable/)).toBeInTheDocument();
  expect(
    screen.getByText("Trial failed; inspect worker logs"),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Next experiments" }),
  ).toBeDisabled();
});

it("renders API failures with a retry action", async () => {
  fetchMock.mockResolvedValue(
    new Response(JSON.stringify({ detail: "History unavailable" }), {
      status: 503,
    }),
  );
  mount();
  expect(await screen.findByText("History unavailable")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
});
