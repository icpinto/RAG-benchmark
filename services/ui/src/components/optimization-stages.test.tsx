import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  OptimizationStages,
  type StageDashboard,
  type StageTrial,
} from "./optimization-stages";

const fetchMock = vi.fn();
const metric = (mean: number | null, reason: string | null = null) => ({
  mean,
  reason,
  valid_count: mean == null ? 0 : 24,
  missing_count: mean == null ? 24 : 0,
});
const config = {
  chunker: { type: "recursive", chunk_size: 512, overlap: 0.1 },
  chunk_units: "characters",
  embedding: { model: "embedding-model" },
  indexers: { dense: { strategy: "dense_hnsw" } },
  retriever: { type: "dense" },
  rewriter: { type: "none" },
  reranker: { type: "none" },
  compressor: { type: "top_n", max_chunks: 5 },
  context: { max_context_chars: 6000 },
  prompt: {
    type: "preset",
    preset_id: "concise_v1",
    template: "Cite evidence: {context} {question}",
  },
  llm: { model: "fixed-model" },
};
function trial(
  id: string,
  phase: StageTrial["phase"],
  score: number,
): StageTrial {
  return {
    id,
    phase,
    split: phase === "confirm" ? "holdout" : "tuning",
    status: "completed",
    display_status: "Completed",
    metrics: {
      "reranked_ndcg@10": metric(score),
      "raw_retrieval_ndcg@10": metric(0.3),
      answer_correctness: metric(score),
      claim_support: metric(0.85),
      citation_support: metric(0.8),
      context_tokens: metric(100),
      total_estimated_cost: metric(null, "unknown pricing"),
      query_latency_seconds: metric(0),
    },
    ranking: {
      rank: 1,
      score,
      paired_count: 24,
      missing_count: 0,
      feasible: true,
      violations: [],
    },
    score_reason: null,
    p95_seconds: 0,
    coverage: { expected: 24, valid: 24, missing: 0, failed: 0 },
    configuration: config,
    roles: ["baseline"],
    selection_reasons: ["Selected on tuning quality"],
    source_trial_id: phase === "retrieval" ? null : "retrieval-best",
    source_answer_trial_id: phase === "confirm" ? "answer-best" : null,
    changes_from_baseline: [
      {
        setting: "prompt.preset_id",
        baseline: "baseline_v1",
        candidate: "concise_v1",
      },
    ],
  };
}
export function dashboard(): StageDashboard {
  return {
    session_id: "session",
    workflow_version: 2,
    stages: {
      retrieval: {
        status: "completed",
        split: "tuning",
        question_count: 24,
        primary_metric: "reranked_ndcg@10",
        trials: [
          trial("retrieval-zero", "retrieval", 0),
          trial("retrieval-best", "retrieval", 0.8),
        ],
      },
      answer: {
        status: "completed",
        split: "tuning",
        question_count: 24,
        primary_metric: "answer_correctness",
        trials: [
          {
            ...trial("answer-best", "answer", 0.9),
            roles: ["selected for holdout"],
          },
          {
            ...trial("answer-errors", "answer", 0.5),
            display_status: "Completed with errors",
            ranking: null,
            score_reason: "insufficient coverage",
            coverage: { expected: 24, valid: 20, missing: 4, failed: 4 },
          },
        ],
        plan: [
          {
            candidate_hash: "a",
            source_trial_id: "retrieval-best",
            variant: "concise_v1",
            scheduled: true,
            trial_id: "answer-best",
            status: "completed",
          },
          {
            candidate_hash: "b",
            source_trial_id: "retrieval-best",
            variant: "reranker:cross_encoder",
            scheduled: false,
            trial_id: null,
            status: "outside trial cap",
          },
        ],
      },
      confirm: {
        status: "completed",
        split: "holdout",
        question_count: 6,
        primary_metric: "answer_correctness",
        trials: [trial("holdout-best", "confirm", 0.95)],
        outcome: "confirmed improvement",
        baseline_retained: false,
        selected_trial_id: "answer-best",
        comparison: {
          compatible: true,
          metrics: [
            {
              metric: "answer_correctness",
              baseline_mean: 0.5,
              candidate_mean: 0.95,
              mean_delta: 0.45,
              ci95: [0.3, 0.6],
              paired_count: 6,
              missing_count: 0,
            },
          ],
        },
        configuration_differences: [
          {
            setting: "prompt.preset_id",
            baseline: "baseline_v1",
            candidate: "concise_v1",
          },
        ],
      },
    },
  };
}
const response = (value: unknown) =>
  Promise.resolve(
    new Response(JSON.stringify(value), {
      headers: { "Content-Type": "application/json" },
    }),
  );
function mount(
  path = "/auto-optimize/session?stage=retrieval",
  status = "completed",
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const router = createMemoryRouter(
    [
      {
        path: "/auto-optimize/:id",
        element: (
          <OptimizationStages
            sessionId="session"
            status={status}
            currentStage={status === "completed" ? "done" : "retrieval"}
          />
        ),
      },
    ],
    { initialEntries: [path] },
  );
  render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return { client, router };
}
beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockReset();
  fetchMock.mockImplementation((path: string) => {
    if (path.endsWith("/stages")) return response(dashboard());
    return response({
      config: {
        pipeline: {
          ingestion: {
            chunkers: [config.chunker],
            embeddings: [config.embedding],
            indexers: config.indexers,
          },
          query: {
            retrievers: [config.retriever],
            rerankers: [config.reranker],
            prompt_builders: [config.prompt],
            context_assemblers: [config.context],
          },
        },
        llm: config.llm,
        runtime: { mode: "end_to_end" },
      },
    });
  });
});

it("shows ordered DSPy recipes, protected capacity, and failures in version six", async () => {
  const data = dashboard();
  data.workflow_version = 6;
  data.stages.prompt = {
    status: "running",
    split: "prompt validation",
    question_count: 6,
    primary_metric: "answer_correctness",
    trials: [],
    parent_answer_trial_id: "answer-best",
    budget_required: { tokens: 4000000, seconds: 1800, cost: null },
    plan: [
      {
        candidate_hash: "first",
        variant: "dspy:mipro_zero_shot_v1",
        source_trial_id: null,
        scheduled: true,
        trial_id: null,
        status: "failed",
        error: "Model timeout",
      },
      {
        candidate_hash: "second",
        variant: "dspy:bootstrap_small_v1",
        source_trial_id: null,
        scheduled: true,
        trial_id: null,
        status: "planned",
      },
    ],
  };
  fetchMock.mockImplementation(() => response(data));
  mount("/auto-optimize/session?stage=prompt", "running");
  expect(await screen.findByRole("tab", { name: /DSPy/ })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  expect(
    await screen.findByText("4,000,000", { exact: false }),
  ).toBeInTheDocument();
  expect(screen.getByText("Model timeout")).toBeInTheDocument();
  expect(screen.getByText("dspy:bootstrap_small_v1")).toBeInTheDocument();
  expect(
    screen.getByText(/No compilation jobs have started yet/),
  ).toBeInTheDocument();
  fireEvent.keyDown(screen.getByRole("tab", { name: /DSPy/ }), { key: "End" });
  expect(screen.getByRole("tab", { name: /Confirmation/ })).toHaveAttribute(
    "aria-selected",
    "true",
  );
});

it("explains a disabled DSPy stage and preserves historical navigation", async () => {
  const data = dashboard();
  data.workflow_version = 6;
  data.stages.prompt = {
    status: "disabled",
    split: "prompt validation",
    question_count: 0,
    primary_metric: "answer_correctness",
    trials: [],
  };
  fetchMock.mockImplementation(() => response(data));
  mount("/auto-optimize/session?stage=prompt");
  expect(
    await screen.findByText(/DSPy is disabled for this run/),
  ).toBeInTheDocument();
});

describe("Optimization stage dashboards", () => {
  it("keeps reranking in answering and removes duplicate evidence tables", async () => {
    fetchMock.mockImplementation((path: string) => {
      if (!path.endsWith("/stages")) {
        return response({
          config: { pipeline: { query: { rerankers: [{ type: "none" }] } } },
        });
      }
      const data = dashboard();
      data.workflow_version = 5;
      data.stages.retrieval.primary_metric = "raw_retrieval_ndcg@10";
      return response(data);
    });
    mount();
    expect(
      await screen.findByRole("figure", {
        name: "Selection score: raw retrieval NDCG@10",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Reranking is disabled/)).toBeInTheDocument();
    expect(
      screen.queryByText("Raw retrieval evidence"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("Raw retrieval versus reranked evidence"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("table", {
        name: "raw-and-reranked table",
      }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText("Retrieval comparison metric"),
    ).not.toBeInTheDocument();
    const retrievalTable = screen.getByRole("table", {
      name: "retrieval-trials table",
    });
    expect(
      within(retrievalTable).queryByRole("columnheader", { name: /reranker/i }),
    ).not.toBeInTheDocument();
    expect(
      within(retrievalTable).queryByText("Reranker:"),
    ).not.toBeInTheDocument();
    const row = (
      await screen.findAllByRole("link", { name: "retrieva" })
    )[0].closest("tr")!;
    fireEvent.click(within(row).getByText("View config"));
    expect(
      await within(row).findByText("Reranker · Disabled during retrieval"),
    ).toBeInTheDocument();
    expect(
      within(row).queryByText("Reranker · Varied here"),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: /Answering/ }));
    expect(
      await screen.findByText(
        /Varied here: reranker, context\/compression and answer prompt/,
      ),
    ).toBeInTheDocument();
    const answerRow = (
      await screen.findByRole("link", { name: "answer-b" })
    ).closest("tr")!;
    fireEvent.click(within(answerRow).getByText("View config"));
    expect(
      await within(answerRow).findByText("Reranker · Varied here"),
    ).toBeInTheDocument();
  });

  it.each([
    "Highest-ranked feasible retrieval trial; sole parent for answer search",
    "Baseline fallback: no feasible ranked retrieval trial was available",
  ])("shows the saved answer handoff reason: %s", async (reason) => {
    fetchMock.mockImplementation(() => {
      const data = dashboard();
      data.stages.answer.plan![0].reason = reason;
      return response(data);
    });
    mount("/auto-optimize/session?stage=answer");
    expect(await screen.findByText(reason)).toBeInTheDocument();
  });

  it("keeps retrieval and answer scores independent, preserves zero, and explains missing scores", async () => {
    const { router } = mount();
    const retrieval = within(
      await screen.findByRole("table", { name: "retrieval-trials table" }),
    );
    expect(retrieval.getAllByText("0.000").length).toBeGreaterThan(0);
    expect(retrieval.getAllByText("0.800").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("tab", { name: /Answering/ }));
    expect(router.state.location.search).toBe("?stage=answer");
    const answers = within(
      await screen.findByRole("table", { name: "answer-trials table" }),
    );
    expect(answers.getAllByText("0.900").length).toBeGreaterThan(0);
    expect(answers.getByText("Completed with errors")).toBeInTheDocument();
    expect(
      answers.getAllByText("insufficient coverage").length,
    ).toBeGreaterThan(0);
    expect(answers.getAllByText("unknown pricing")).toHaveLength(2);
    fireEvent.click(screen.getByText(/Search coverage:/));
    expect(screen.getByText("reranker:cross_encoder")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: /Retrieval/ }));
    expect(
      within(
        screen.getByRole("table", { name: "retrieval-trials table" }),
      ).getAllByText("0.800").length,
    ).toBeGreaterThan(0);
  });

  it("deep-links to answering, lazily shows settings, and traces the source retrieval trial", async () => {
    const { router } = mount("/auto-optimize/session?stage=answer");
    const row = (await screen.findByRole("link", { name: "answer-b" })).closest(
      "tr",
    )!;
    fireEvent.click(within(row).getByText("View config"));
    await within(row).findByText("Chunker · Inherited");
    expect(
      within(row).getByText("Prompt settings · Varied here"),
    ).toBeInTheDocument();
    expect(
      within(row).getByText("Selected on tuning quality"),
    ).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.filter(([path]) => path.endsWith("/config")),
    ).toHaveLength(1);
    fireEvent.click(within(row).getAllByRole("link", { name: "retrieva" })[0]);
    expect(router.state.location.search).toBe("?stage=retrieval");
    expect(router.state.location.hash).toBe("#trial-retrieval-best");
    const rows = within(
      await screen.findByRole("table", { name: "retrieval-trials table" }),
    ).getAllByRole("row");
    expect(within(rows[1]).getByRole("link")).toHaveTextContent("retrieva");
    expect(rows[1]).toContainElement(
      document.getElementById("trial-retrieval-best"),
    );
  });

  it("shows holdout comparison and an accessible confidence interval without using tuning scores", async () => {
    mount("/auto-optimize/session?stage=confirm");
    await screen.findByText(/Holdout correctness delta:/);
    expect(
      screen.getByRole("img", {
        name: "95% confidence interval 0.300 to 0.600",
      }),
    ).toBeInTheDocument();
    const comparison = within(
      screen.getByRole("table", { name: "holdout-comparison table" }),
    );
    expect(comparison.getByText("0.95")).toBeInTheDocument();
    expect(comparison.getByText("0.45")).toBeInTheDocument();
    expect(
      screen.queryByRole("table", { name: "answer-trials table" }),
    ).not.toBeInTheDocument();
    fireEvent.click(
      screen.getByText("Baseline and candidate configuration differences"),
    );
    expect(
      screen.getByRole("table", {
        name: "holdout-configuration-differences table",
      }),
    ).toBeInTheDocument();
  });

  it("keeps acceptance evidence without repeating the recommendation summary", async () => {
    const data = dashboard();
    data.stages.confirm.outcome = "promising candidate";
    data.stages.confirm.baseline_retained = false;
    data.stages.confirm.acceptance = {
      outcome: "promising_candidate",
      recommended_source: "candidate",
      reason_codes: ["statistical_confirmation"],
      checks: [
        {
          code: "paired_coverage",
          label: "At least 95% paired scores",
          status: "pass",
          measured: { paired: 6, expected: 6, fraction: 1 },
          threshold: 0.95,
          required: true,
        },
        {
          code: "statistical_confirmation",
          label: "Improvement is statistically confirmed",
          status: "fail",
          measured: -0.02,
          threshold: 0,
          required: false,
        },
      ],
    };
    data.stages.confirm.explanation = {
      text: "The candidate is promising, but improvement remains uncertain.",
      source: "llm",
      prompt_version: "v1",
    };
    fetchMock.mockImplementation(() => response(data));
    mount("/auto-optimize/session?stage=confirm");
    await screen.findByText(/6 \/ 6 \(100.0%\)/);
    expect(
      screen.getByText(/Not met: Improvement is statistically confirmed/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Promising candidate")).not.toBeInTheDocument();
    expect(screen.queryByText(/Candidate recommended/)).not.toBeInTheDocument();
    expect(
      screen.queryByText(data.stages.confirm.explanation.text),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(/Baseline recommendation retained/),
    ).not.toBeInTheDocument();
  });

  it("preserves the chosen stage across polling and refreshes metrics", async () => {
    let updated = false;
    fetchMock.mockImplementation(() => {
      const data = dashboard();
      if (updated)
        data.stages.answer.trials[0].display_status = "Completed with errors";
      return response(data);
    });
    const { router, client } = mount(
      "/auto-optimize/session?stage=answer",
      "running",
    );
    await screen.findByRole("table", { name: "answer-trials table" });
    updated = true;
    await act(() =>
      client.invalidateQueries({ queryKey: ["optimization-stages"] }),
    );
    expect(router.state.location.search).toBe("?stage=answer");
    await waitFor(() =>
      expect(screen.getAllByText("Completed with errors")).toHaveLength(2),
    );
  });

  it("keeps stage fetch errors local and supports retry", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Stage results unavailable" }), {
        status: 503,
      }),
    );
    mount();
    await screen.findByText("Stage results unavailable");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByRole("table", { name: "retrieval-trials table" });
  });

  it("supports keyboard stage navigation", async () => {
    const { router } = mount();
    await screen.findByRole("table", { name: "retrieval-trials table" });
    fireEvent.keyDown(screen.getByRole("tab", { name: /Retrieval/ }), {
      key: "ArrowRight",
    });
    expect(router.state.location.search).toBe("?stage=answer");
    expect(screen.getByRole("tab", { name: /Answering/ })).toHaveFocus();
  });

  it("identifies historical workflows without claiming new stage search was used", async () => {
    fetchMock.mockImplementation((path: string) => {
      if (path.endsWith("/stages")) {
        const data = dashboard();
        data.workflow_version = 1;
        return response(data);
      }
      return response({
        config: {
          pipeline: {
            ingestion: { chunkers: [{ type: "fixed", chunk_size: 256 }] },
          },
        },
      });
    });
    mount();
    await screen.findByText(/Historical workflow:/);
    const row = screen
      .getAllByRole("link", { name: "retrieva" })[0]
      .closest("tr")!;
    fireEvent.click(within(row).getByText("View config"));
    await within(row).findByText("Chunker");
    expect(
      within(row).queryByText("Chunker · Varied here"),
    ).not.toBeInTheDocument();
  });
});
