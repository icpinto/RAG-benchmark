import {
  render,
  screen,
  fireEvent,
  waitFor,
  within,
  act,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AutoOptimize from "./auto-optimize";
import { DraftProvider } from "../hooks/draft";

const fetchMock = vi.fn();
const response = (data: unknown) =>
  Promise.resolve(
    new Response(JSON.stringify(data), {
      headers: { "Content-Type": "application/json" },
    }),
  );
const dataset = {
  version: 1,
  frozen: 0,
  payload: [
    {
      question_id: "q1",
      question: "What is the refund window?",
      expected_answer: "Fourteen days.",
      category: "fact",
      split: "unassigned",
      human_reviewed: false,
      automated_validated: true,
      evidence: [
        {
          doc_id: "policy.txt",
          quote: "Refunds within fourteen days.",
          start_char: 0,
          end_char: 29,
        },
      ],
    },
  ],
};
const session = {
  id: "session",
  status: "paused",
  stage: "freeze",
  budget: { tokens: 100, active_seconds: 12, cost: null },
  progress: {},
};
const savedConfiguration = {
  request: {
    config_path: "saved-template.yaml",
    source_paths: ["captured-policy.txt"],
    urls: ["https://example.test/policy"],
    planner: { model: "saved-planner", provider: "ollama" },
    synthesizer: { model: "saved-synthesizer", provider: "ollama" },
    verifier: { model: "saved-verifier", provider: "ollama" },
    judge: { model: "saved-judge", provider: "ollama" },
    limits: {
      target_questions: 75,
      retrieval_trials: 12,
      answer_trials: 3,
      active_seconds: 3600,
      max_tokens: 123456,
      max_cost: null,
      max_query_p95_seconds: 8,
      max_query_cost: null,
    },
    review_before_freeze: true,
    allow_hosted: false,
    allowed_components: { retrievers: ["dense", "bm25"] },
  },
  template: { llm: { model: "saved-answer", provider: "ollama" } },
};
const trialConfig = {
  pipeline: {
    ingestion: {
      chunkers: [{ type: "recursive", chunk_size: 512, overlap: 0.1 }],
      embeddings: [
        { provider: "sentence_transformers", model: "saved-embedding" },
      ],
    },
    query: {
      retrievers: [{ type: "dense" }],
      rewriters: [{ type: "multi_query" }],
      rerankers: [{ type: "cross_encoder" }],
      compressors: [{ type: "top_n" }],
      context_assemblers: [
        { type: "basic", max_chunks: 5, max_context_chars: 12000 },
      ],
    },
  },
  runtime: { mode: "retrieval_only" },
  metrics: { enabled: ["retrieval_metrics"] },
};
const trial = (id: string, status: string) => ({
  id,
  status,
  phase: "retrieval",
  split: "tuning",
});

function stagePayload(trials: ReturnType<typeof trial>[] = []) {
  return {
    session_id: "session",
    workflow_version: 1,
    stages: Object.fromEntries(
      ["retrieval", "answer", "confirm"].map((phase) => [
        phase,
        {
          status: "running",
          split: phase === "confirm" ? "holdout" : "tuning",
          question_count: 30,
          primary_metric:
            phase === "retrieval" ? "reranked_ndcg@10" : "answer_correctness",
          trials: trials
            .filter((t) => t.phase === phase)
            .map((t) => ({
              ...t,
              display_status: t.status,
              metrics: {},
              ranking: null,
              score_reason: "pending",
              p95_seconds: null,
              coverage: { expected: 30, valid: 0, missing: 30, failed: 0 },
              configuration: null,
              roles: [],
              selection_reasons: [],
              source_trial_id: null,
              source_answer_trial_id: null,
              changes_from_baseline: [],
            })),
        },
      ]),
    ),
  };
}

function overrideFetch(
  handler: (path: string) => ReturnType<typeof response> | undefined,
) {
  const fallback = fetchMock.getMockImplementation()!;
  fetchMock.mockImplementation((path: string, options?: RequestInit) => {
    const direct = handler(path);
    if (direct) return direct;
    if (path.endsWith("/stages")) {
      const trials = handler(path.replace(/\/stages$/, "/trials"));
      if (trials)
        return trials.then(async (r) => response(stagePayload(await r.json())));
    }
    return fallback(path, options);
  });
}

function card(title: string) {
  return within(
    screen.getByRole("heading", { name: title }).closest("section")!,
  );
}

function mount(path = "/auto-optimize") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const router = createMemoryRouter(
    [
      {
        path: "/auto-optimize",
        element: (
          <DraftProvider>
            <AutoOptimize />
          </DraftProvider>
        ),
      },
      {
        path: "/auto-optimize/new",
        element: (
          <DraftProvider>
            <AutoOptimize />
          </DraftProvider>
        ),
      },
      {
        path: "/auto-optimize/:sessionId",
        element: (
          <DraftProvider>
            <AutoOptimize />
          </DraftProvider>
        ),
      },
    ],
    { initialEntries: [path] },
  );
  const view = render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return { ...view, client, router };
}
beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockReset();
  fetchMock.mockImplementation((path: string, options?: RequestInit) => {
    if (path === "/api/configs") return response({ files: ["default.yaml"] });
    if (path === "/api/data/files")
      return response({ files: [{ name: "policy.txt", path: "policy.txt" }] });
    if (path === "/api/optimizations" && options?.method === "POST")
      return response({ id: "session" });
    if (path === "/api/optimizations") return response([session]);
    if (path.endsWith("/configuration")) return response(savedConfiguration);
    if (path.endsWith("/config")) return response({ config: trialConfig });
    if (path.endsWith("/dataset")) return response(dataset);
    if (path.endsWith("/stages")) return response(stagePayload());
    if (path.endsWith("/trials") || path.endsWith("/events"))
      return response([]);
    if (path.endsWith("/recommendation")) return response(null);
    if (options?.method === "POST") return response({ status: "queued" });
    return response(session);
  });
});
describe("Auto Optimize", () => {
  it.each([
    ["confirmed_improvement", "Confirmed improvement"],
    ["promising_candidate", "Promising candidate"],
    ["baseline_retained", "Baseline retained"],
  ])(
    "renders the saved %s decision and its explanation",
    async (outcome, label) => {
      overrideFetch((path) => {
        if (path.endsWith("/recommendation"))
          return response({
            yaml: "experiment:\n  name: recommended-candidate\n",
            confirmed_improvement: outcome === "confirmed_improvement",
            baseline_retained: outcome === "baseline_retained",
            limitations: [],
            acceptance: {
              outcome,
              recommended_source:
                outcome === "baseline_retained" ? "baseline" : "candidate",
              reason_codes: ["statistical_confirmation"],
              checks: [
                {
                  code: "paired_coverage",
                  label: "At least 95% paired scores",
                  status: "pass",
                  measured: { paired: 19, expected: 20, fraction: 0.95 },
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
            },
            explanation: {
              text: "<strong>Recorded explanation</strong>",
              source: "fallback",
              prompt_version: "v1",
            },
          });
      });
      mount("/auto-optimize/session");
      await screen.findByRole("heading", { name: "Recommendation" });
      const recommendation = card("Recommendation");
      expect(recommendation.getByText(label)).toBeInTheDocument();
      const text = recommendation.getByText(
        "<strong>Recorded explanation</strong>",
      );
      expect(text.querySelector("strong")).toBeNull();
      expect(
        recommendation.getByText(/19 \/ 20 \(95.0%\)/),
      ).toBeInTheDocument();
      expect(
        recommendation.getByText(/Model explanation unavailable/),
      ).toBeInTheDocument();
      expect(
        recommendation.getByText(
          /Not met: Improvement is statistically confirmed/,
        ),
      ).toBeInTheDocument();
      expect(
        recommendation.getByRole("button", { name: "Open in Benchmark Setup" }),
      ).toBeEnabled();
    },
  );

  it("keeps historical recommendation rendering without new fields", async () => {
    overrideFetch((path) =>
      path.endsWith("/recommendation")
        ? response({
            yaml: "experiment: {}",
            confirmed_improvement: false,
            baseline_retained: true,
            limitations: ["Historical report"],
          })
        : undefined,
    );
    mount("/auto-optimize/session");
    await screen.findByText(
      "Baseline retained; candidate improvement has not been confirmed.",
    );
    expect(
      card("Recommendation").getByText("Historical report"),
    ).toBeInTheDocument();
  });

  it("starts frozen runs collapsed and keeps evaluation data inside configuration", async () => {
    overrideFetch((path) =>
      path.endsWith("/dataset")
        ? response({ ...dataset, frozen: 1 })
        : undefined,
    );
    mount("/auto-optimize/session");
    const heading = await screen.findByRole("heading", {
      name: "Run configuration",
    });
    const disclosure = heading.closest("details")!;
    await screen.findByText(/saved-template.yaml.*1 questions/);
    expect(disclosure.open).toBe(false);
    expect(
      screen.queryByRole("heading", { name: /Synthetic evaluation dataset/ }),
    ).not.toBeInTheDocument();
    expect(
      within(disclosure).getByRole("region", {
        name: "Evaluation dataset",
        hidden: true,
      }),
    ).toBeInTheDocument();
    fireEvent.click(heading.closest("summary")!);
    await waitFor(() => expect(disclosure.open).toBe(true));
  });

  it("opens when review becomes required and respects manual collapse across updates", async () => {
    overrideFetch((path) => {
      if (path === "/api/optimizations/session")
        return response({ ...session, status: "running" });
      if (path.endsWith("/dataset")) return response({ ...dataset, frozen: 1 });
    });
    const { client } = mount("/auto-optimize/session");
    const heading = await screen.findByRole("heading", {
      name: "Run configuration",
    });
    const disclosure = heading.closest("details")!;
    await screen.findByText(/saved-template.yaml.*Frozen/);
    expect(disclosure.open).toBe(false);
    act(() => {
      client.setQueryData(["optimization", "session"], session);
      client.setQueryData(["optimization-dataset", "session"], dataset);
    });
    await waitFor(() => expect(disclosure.open).toBe(true));
    expect(screen.getByText("Review required")).toBeVisible();
    fireEvent.click(heading.closest("summary")!);
    await waitFor(() => expect(disclosure.open).toBe(false));
    act(() =>
      client.setQueryData(["optimization-dataset", "session"], {
        ...dataset,
        version: 2,
      }),
    );
    expect(disclosure.open).toBe(false);
  });

  it("keeps dataset review available when saved configuration fails", async () => {
    overrideFetch((path) =>
      path.endsWith("/configuration")
        ? Promise.resolve(
            new Response(JSON.stringify({ detail: "Settings unavailable" }), {
              status: 503,
            }),
          )
        : undefined,
    );
    mount("/auto-optimize/session");
    await screen.findByText("Settings unavailable");
    const evaluation = await screen.findByRole("region", {
      name: "Evaluation dataset",
    });
    expect(
      within(evaluation).getByLabelText("Draft evaluation questions JSON"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Approve draft and resume" }),
    ).toBeEnabled();
  });

  it("annotates running configurations using their saved stage and workflow", async () => {
    overrideFetch((path) => {
      if (path.endsWith("/stages"))
        return response({ ...stagePayload(), workflow_version: 5 });
      if (path.endsWith("/trials"))
        return response([
          trial("raw-running", "running"),
          { ...trial("answer-running", "running"), phase: "answer" },
          { ...trial("confirm-running", "running"), phase: "confirm" },
        ]);
    });
    mount("/auto-optimize/session");
    const raw = within(
      await screen.findByRole("article", { name: "Running trial raw-running" }),
    );
    await raw.findByText("Reranker · Disabled during retrieval");
    const answer = within(
      screen.getByRole("article", { name: "Running trial answer-running" }),
    );
    await answer.findByText("Reranker · Varied here");
    expect(answer.getByText("Retriever · Inherited")).toBeInTheDocument();
    const confirm = within(
      screen.getByRole("article", { name: "Running trial confirm-running" }),
    );
    await confirm.findByText("Reranker · Fixed for confirmation");
  });

  it("sends selected DSPy recipes to Auto Optimize", async () => {
    mount("/auto-optimize/new");
    fireEvent.click(
      screen.getByText("Model roles and advanced search settings", {
        selector: "summary",
      }),
    );
    fireEvent.click(await screen.findByLabelText("Enable DSPy compilation"));
    fireEvent.click(screen.getByLabelText("gepa_light_v1"));
    fireEvent.click(screen.getByRole("button", { name: "Start optimization" }));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, options]) =>
            url === "/api/optimizations" && options?.method === "POST",
        ),
      ).toBe(true),
    );
    const request = fetchMock.mock.calls.find(
      ([url, options]) =>
        url === "/api/optimizations" && options?.method === "POST",
    );
    expect(
      JSON.parse(request![1].body).prompt_optimization.configs.map(
        (c: { preset: string }) => c.preset,
      ),
    ).toEqual(["mipro_zero_shot_v1", "gepa_light_v1"]);
  });
  it("opens history by default with newest-first metadata and detail links", async () => {
    const created = 1_790_000_000;
    overrideFetch((path) =>
      path === "/api/optimizations"
        ? response([
            { ...session, id: "older", created_at: created - 3600 },
            {
              ...session,
              id: "newer",
              status: "completed",
              created_at: created,
              budget: { tokens: 12345, active_seconds: 90, cost: 0.125 },
            },
          ])
        : undefined,
    );
    mount();
    const table = within(
      await screen.findByRole("table", { name: "optimization runs table" }),
    );
    expect(screen.queryByText("Start an optimization")).not.toBeInTheDocument();
    expect(screen.getByText(/newest 100/)).toBeInTheDocument();
    const rows = table.getAllByRole("row");
    const newest = within(rows[1]);
    expect(newest.getByRole("link", { name: "newer" })).toHaveAttribute(
      "href",
      "/auto-optimize/newer",
    );
    expect(newest.getByText("completed")).toBeInTheDocument();
    expect(newest.getByText("12,345")).toBeInTheDocument();
    expect(newest.getByText("1.5")).toBeInTheDocument();
    expect(newest.getByText("0.1250")).toBeInTheDocument();
    expect(rows[1].querySelector("time")).toHaveAttribute(
      "datetime",
      new Date(created * 1000).toISOString(),
    );
    expect(within(rows[2]).getByText("Unknown pricing")).toBeInTheDocument();
    fireEvent.click(
      newest.getByRole("link", { name: "View details for run newer" }),
    );
    await screen.findByRole("heading", { name: "Run configuration" });
    expect(
      screen.getByRole("link", { name: "Back to run history" }),
    ).toHaveAttribute("href", "/auto-optimize");
  });

  it("searches, sorts and pages through history", async () => {
    overrideFetch((path) =>
      path === "/api/optimizations"
        ? response(
            Array.from({ length: 12 }, (_, index) => ({
              ...session,
              id: `run-${index}`,
              created_at: 1_790_000_000 + index,
            })),
          )
        : undefined,
    );
    mount();
    const table = within(await screen.findByRole("table"));
    expect(table.getAllByRole("row")).toHaveLength(11);
    expect(screen.getByLabelText("Rows per page")).toHaveValue("10");
    expect(table.queryByRole("link", { name: "run-0" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(table.getByRole("link", { name: "run-0" })).toBeVisible();
    fireEvent.change(screen.getByLabelText("Search optimization runs"), {
      target: { value: "run-11" },
    });
    await waitFor(() => expect(table.getAllByRole("row")).toHaveLength(2));
    expect(table.getByRole("link", { name: "run-11" })).toBeVisible();
    fireEvent.change(screen.getByLabelText("Search optimization runs"), {
      target: { value: "no-such-run" },
    });
    await screen.findByText("No matching rows.");
    fireEvent.change(screen.getByLabelText("Search optimization runs"), {
      target: { value: "" },
    });
    fireEvent.click(table.getByRole("button", { name: "created at" }));
    expect(
      table.getByRole("button", { name: "created at" }).closest("th"),
    ).toHaveAttribute("aria-sort", "descending");
    fireEvent.click(table.getByRole("button", { name: "created at" }));
    await waitFor(() =>
      expect(
        within(table.getAllByRole("row")[1]).getByRole("link", {
          name: "run-0",
        }),
      ).toBeVisible(),
    );
  });

  it("shows loading, error, retry and empty history states separately", async () => {
    let resolveHistory!: (value: Response) => void;
    overrideFetch((path) =>
      path === "/api/optimizations"
        ? new Promise<Response>((resolve) => {
            resolveHistory = resolve;
          })
        : undefined,
    );
    mount();
    expect(screen.getByText("Loading run history...")).toBeInTheDocument();
    expect(screen.queryByText("No optimizations yet")).not.toBeInTheDocument();
    await act(async () =>
      resolveHistory(
        new Response(JSON.stringify({ detail: "History unavailable" }), {
          status: 503,
        }),
      ),
    );
    await screen.findByText("History unavailable");
    expect(screen.queryByText("No optimizations yet")).not.toBeInTheDocument();
    overrideFetch((path) =>
      path === "/api/optimizations" ? response([]) : undefined,
    );
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByText("No optimizations yet");
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("link", { name: "New optimization" }));
    await screen.findByRole("heading", { name: "Start an optimization" });
    expect(
      fetchMock.mock.calls.some(([path]) => path === "/api/optimizations/new"),
    ).toBe(false);
  });

  it("refreshes run status in history every five seconds", async () => {
    let status = "running";
    overrideFetch((path) =>
      path === "/api/optimizations"
        ? response([{ ...session, status }])
        : undefined,
    );
    mount();
    await screen.findByText("running", { selector: ".status-badge" });
    status = "completed";
    await screen.findByText(
      "completed",
      { selector: ".status-badge" },
      { timeout: 6500 },
    );
  }, 8000);

  it.each(["completed", "failed", "cancelled", "budget_exhausted"])(
    "opens a historical %s run with saved details and no live controls",
    async (status) => {
      const created = 1_790_000_000;
      overrideFetch((path) =>
        path === "/api/optimizations/session"
          ? response({
              ...session,
              status,
              created_at: created,
              updated_at: created + 60,
            })
          : undefined,
      );
      mount("/auto-optimize/session");
      await screen.findByRole("heading", { name: `Optimization · ${status}` });
      const summary = card(`Optimization · ${status}`);
      expect(
        summary.getByText("session", { selector: "dd" }),
      ).toBeInTheDocument();
      expect(
        summary.getByText(new Date(created * 1000).toLocaleString()),
      ).toHaveAttribute("datetime", new Date(created * 1000).toISOString());
      expect(
        summary.getByText(new Date((created + 60) * 1000).toLocaleString()),
      ).toBeInTheDocument();
      await screen.findByText("saved-template.yaml");
      expect(screen.queryByRole("button", { name: "Pause" })).toBeNull();
      expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
      expect(screen.queryByRole("button", { name: /resume/i })).toBeNull();
      expect(
        screen.getByRole("button", { name: "Start a new version" }),
      ).toBeEnabled();
    },
  );

  it("keeps missing and invalid historical dates explicit", async () => {
    overrideFetch((path) =>
      path === "/api/optimizations"
        ? response([
            { ...session },
            { ...session, id: "invalid-date", created_at: 1e30 },
          ])
        : undefined,
    );
    mount();
    await screen.findByRole("table");
    expect(screen.getAllByText("Date unavailable")).toHaveLength(2);
  });

  it("shows a retryable detail error without hiding history navigation", async () => {
    let failed = true;
    overrideFetch((path) =>
      path === "/api/optimizations/missing"
        ? failed
          ? Promise.resolve(
              new Response(
                JSON.stringify({ detail: "Optimization session not found" }),
                { status: 404 },
              ),
            )
          : response(session)
        : undefined,
    );
    mount("/auto-optimize/missing");
    await screen.findByText("Optimization session not found");
    expect(
      screen.getByRole("link", { name: "Back to run history" }),
    ).toBeVisible();
    failed = false;
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByRole("heading", { name: "Run configuration" });
  });

  it("clears unsaved dataset edits when switching runs", async () => {
    const { router } = mount("/auto-optimize/session");
    const draft = await screen.findByLabelText(
      "Draft evaluation questions JSON",
    );
    fireEvent.change(draft, {
      target: { value: "unsaved edits from previous run" },
    });
    await act(() => router.navigate("/auto-optimize/other"));
    const nextDraft = await screen.findByLabelText(
      "Draft evaluation questions JSON",
    );
    expect(nextDraft).not.toHaveValue("unsaved edits from previous run");
    expect(nextDraft).toHaveValue(JSON.stringify(dataset.payload, null, 2));
    expect(
      screen.getByRole("button", { name: "Approve draft and resume" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Save dataset edits" }),
    ).toBeDisabled();
  });

  it("submits bounded defaults and opens the captured session", async () => {
    mount("/auto-optimize/new");
    fireEvent.change(screen.getByLabelText("Task guidance (optional)"), {
      target: { value: "Support policy questions" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start optimization" }));
    await screen.findByText("Optimization · paused");
    const request = fetchMock.mock.calls.find(
      ([path, options]) =>
        path === "/api/optimizations" && options?.method === "POST",
    );
    const payload = JSON.parse(request![1].body);
    expect(payload.limits).toMatchObject({
      target_questions: 100,
      retrieval_trials: 20,
      answer_trials: 5,
      active_seconds: 7200,
    });
    expect(payload.task).toBe("Support policy questions");
    expect(payload.allow_hosted).toBe(false);
    expect(payload.evaluation_dataset_mode).toBe("reuse_if_compatible");
    expect(payload.planner.provider).toBe("ollama");
    for (const role of ["planner", "synthesizer", "verifier", "judge"]) {
      expect(payload[role]).toMatchObject({
        model: "qwen2.5:32b",
        base_url: "http://100.110.250.66:11434",
      });
    }
  });
  it("allows generating fresh evaluation questions", async () => {
    mount("/auto-optimize/new");
    fireEvent.change(screen.getByLabelText("Evaluation questions"), {
      target: { value: "generate_new" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start optimization" }));
    await waitFor(() => {
      const request = fetchMock.mock.calls.find(
        ([path, options]) =>
          path === "/api/optimizations" && options?.method === "POST",
      );
      expect(JSON.parse(request![1].body).evaluation_dataset_mode).toBe(
        "generate_new",
      );
    });
  });
  it("shows reused questions and their origin without an editable draft", async () => {
    overrideFetch((path) =>
      path.endsWith("/dataset")
        ? response({
            ...dataset,
            frozen: 1,
            evaluation_dataset: {
              status: "reused",
              dataset_id: "copy",
              source_dataset_id: "original",
              source_session_id: "origin-session",
              question_count: 1,
            },
          })
        : undefined,
    );
    mount("/auto-optimize/session");
    const origin = await screen.findByRole("link", {
      name: "originating experiment",
    });
    expect(origin).toHaveAttribute("href", "/auto-optimize/origin-session");
    expect(screen.getByText(/Reused evaluation questions/)).toBeInTheDocument();
    expect(screen.getAllByText(/1 questions/).length).toBeGreaterThan(0);
    expect(screen.queryByText("Edit draft questions")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Resume optimization" }),
    ).toBeEnabled();
  });
  it("displays evidence and supports an explicit review resume", async () => {
    mount("/auto-optimize/session");
    await screen.findByRole("heading", {
      name: "1. What is the refund window?",
    });
    expect(
      screen.getByText("Refunds within fourteen days.", { exact: true }),
    ).toBeInTheDocument();
    expect(screen.getByText("Unknown pricing")).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Approve draft and resume" }),
    );
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/optimizations/session/resume",
        expect.objectContaining({ method: "POST" }),
      ),
    );
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Approve draft and resume" }),
      ).toBeEnabled(),
    );
    expect(
      fetchMock.mock.calls.filter(([path]) => path.endsWith("/configuration")),
    ).toHaveLength(1);
  });

  it("shows saved settings and read-only YAML on a revisited run", async () => {
    mount("/auto-optimize/session");
    await screen.findByText("saved-template.yaml");
    const run = card("Run configuration");
    expect(run.getByText("captured-policy.txt")).toBeInTheDocument();
    expect(run.getByText("https://example.test/policy")).toBeInTheDocument();
    for (const model of [
      "planner",
      "synthesizer",
      "verifier",
      "judge",
      "answer",
    ]) {
      expect(
        run.getByText(new RegExp(`model: saved-${model}`), { selector: "dd" }),
      ).toBeInTheDocument();
    }
    expect(run.getByText("123456")).toBeInTheDocument();
    expect(run.getByText("60")).toBeInTheDocument();
    expect(run.getByText("retrievers: dense; bm25")).toBeInTheDocument();
    fireEvent.click(
      run.getByText("Saved request YAML", { selector: "summary" }),
    );
    expect(run.getByLabelText("Saved request YAML")).toHaveTextContent(
      "target_questions: 75",
    );
    fireEvent.click(
      run.getByText("Saved template YAML", { selector: "summary" }),
    );
    expect(run.getByLabelText("Saved template YAML")).toHaveTextContent(
      "model: saved-answer",
    );
    expect(
      run
        .getByRole("region", { name: "Sources and guidance" })
        .querySelector("input, textarea, select"),
    ).toBeNull();
    expect(
      screen.getByText("No trial is currently running."),
    ).toBeInTheDocument();
  });

  it("shows every running config and lazily loads other trial configs using the same cache", async () => {
    overrideFetch((path) =>
      path.endsWith("/trials")
        ? response([
            trial("running-a", "running"),
            trial("running-b", "running"),
            trial("queued-c", "queued"),
            trial("finished-d", "completed"),
            trial("paused-e", "paused"),
            trial("failed-f", "failed"),
          ])
        : undefined,
    );
    mount("/auto-optimize/session");
    const runningA = within(
      await screen.findByRole("article", { name: "Running trial running-a" }),
    );
    await runningA.findByText("type: recursive, chunk_size: 512, overlap: 0.1");
    const runningB = within(
      screen.getByRole("article", { name: "Running trial running-b" }),
    );
    await runningB.findByText(
      "provider: sentence_transformers, model: saved-embedding",
    );
    expect(
      runningA.getByText(
        "type: basic, max_chunks: 5, max_context_chars: 12000",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("article", { name: "Running trial queued-c" }),
    ).not.toBeInTheDocument();
    const configCalls = () =>
      fetchMock.mock.calls.filter(([path]) => path.endsWith("/config"));
    expect(configCalls()).toHaveLength(2);
    const runningRow = within(
      screen.getAllByRole("link", { name: "running-" })[0].closest("tr")!,
    );
    fireEvent.click(runningRow.getByText("View config"));
    await runningRow.findByText("retrieval_only");
    expect(configCalls()).toHaveLength(2);
    for (const id of ["queued-c", "finished-d", "paused-e", "failed-f"]) {
      const row = within(
        screen.getByRole("link", { name: id.slice(0, 8) }).closest("tr")!,
      );
      fireEvent.click(row.getByText("View config"));
      await row.findByText("retrieval_only");
      fireEvent.click(
        row.getByText("Full trial configuration YAML", { selector: "summary" }),
      );
      expect(
        row.getByLabelText("Full trial configuration YAML"),
      ).toHaveTextContent("retrieval_metrics");
    }
    expect(configCalls()).toHaveLength(6);
  });

  it("updates running cards as trials start and finish through polling", async () => {
    let status = "queued";
    overrideFetch((path) =>
      path.endsWith("/trials")
        ? response([trial("polled-trial", status)])
        : undefined,
    );
    mount("/auto-optimize/session");
    await screen.findByText("No trial is currently running.");
    status = "running";
    await screen.findByRole(
      "article",
      { name: "Running trial polled-trial" },
      { timeout: 4500 },
    );
    await screen.findByText("retrieval_only");
    status = "completed";
    await screen.findByText(
      "No trial is currently running.",
      {},
      { timeout: 4500 },
    );
    expect(
      screen.queryByRole("article", { name: "Running trial polled-trial" }),
    ).not.toBeInTheDocument();
    expect(
      fetchMock.mock.calls.filter(([path]) => path.endsWith("/config")),
    ).toHaveLength(1);
  }, 12000);

  it("keeps configuration errors local and retries failed requests", async () => {
    let failed = true;
    overrideFetch((path) => {
      if (path.endsWith("/trials"))
        return response([trial("retry-trial", "running")]);
      if (
        failed &&
        (path.endsWith("/configuration") || path.endsWith("/config"))
      ) {
        return Promise.resolve(
          new Response(
            JSON.stringify({ detail: "Configuration service unavailable" }),
            { status: 503 },
          ),
        );
      }
    });
    mount("/auto-optimize/session");
    await screen.findAllByText("Configuration service unavailable");
    expect(
      screen.getByRole("heading", { name: "Optimization · paused" }),
    ).toBeInTheDocument();
    failed = false;
    fireEvent.click(
      card("Run configuration").getByRole("button", { name: "Try again" }),
    );
    await screen.findByText("saved-template.yaml");
    const running = within(
      screen.getByRole("article", { name: "Running trial retry-trial" }),
    );
    fireEvent.click(await running.findByRole("button", { name: "Try again" }));
    await running.findByText("retrieval_only");
  });

  it("refreshes trials one final time when the session finishes", async () => {
    let finished = false;
    overrideFetch((path) =>
      path.endsWith("/trials")
        ? response([trial("final-trial", finished ? "completed" : "running")])
        : undefined,
    );
    const { client } = mount("/auto-optimize/session");
    await screen.findByRole("article", { name: "Running trial final-trial" });
    await screen.findByText("retrieval_only");
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(([path]) => path.endsWith("/stages")),
      ).toHaveLength(1),
    );
    finished = true;
    act(() =>
      client.setQueryData(["optimization", "session"], {
        ...session,
        status: "completed",
        stage: "done",
      }),
    );
    await screen.findByText("No trial is currently running.");
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(([path]) => path.endsWith("/stages")),
      ).toHaveLength(2),
    );
    expect(
      screen.queryByRole("article", { name: "Running trial final-trial" }),
    ).not.toBeInTheDocument();
    expect(
      fetchMock.mock.calls.filter(([path]) => path.endsWith("/config")),
    ).toHaveLength(1);
    expect(
      fetchMock.mock.calls.filter(([path]) => path.endsWith("/configuration")),
    ).toHaveLength(1);
  });

  it("shows loading and unavailable configuration states", async () => {
    let resolveRun!: (value: Response) => void;
    let resolveTrial!: (value: Response) => void;
    overrideFetch((path) => {
      if (path.endsWith("/configuration"))
        return new Promise<Response>((resolve) => {
          resolveRun = resolve;
        });
      if (path.endsWith("/config"))
        return new Promise<Response>((resolve) => {
          resolveTrial = resolve;
        });
      if (path.endsWith("/trials"))
        return response([trial("empty-trial", "running")]);
    });
    mount("/auto-optimize/session");
    await screen.findByText("Loading run configuration...");
    await screen.findByText("Loading trial configuration...");
    await act(async () => {
      resolveRun(await response({ request: null, template: null }));
      resolveTrial(await response({ config: null }));
    });
    await screen.findByText("Run configuration is unavailable.");
    await screen.findByText("Trial configuration is unavailable.");
  });

  it("loads the next session without retaining the previous configs", async () => {
    overrideFetch((path) => {
      if (path.endsWith("/trials"))
        return response([trial("shared-id", "running")]);
      if (path.startsWith("/api/optimizations/other/")) {
        if (path.endsWith("/configuration"))
          return response({
            ...savedConfiguration,
            request: {
              ...savedConfiguration.request,
              config_path: "other-template.yaml",
            },
          });
        if (path.endsWith("/config"))
          return response({ config: { runtime: { mode: "end_to_end" } } });
      }
    });
    const { router } = mount("/auto-optimize/session");
    await screen.findByText("saved-template.yaml");
    await screen.findByText("retrieval_only");
    await act(() => router.navigate("/auto-optimize/other"));
    await screen.findByText("other-template.yaml");
    await screen.findByText("end_to_end");
    expect(screen.queryByText("saved-template.yaml")).not.toBeInTheDocument();
    expect(screen.queryByText("retrieval_only")).not.toBeInTheDocument();
  });
});
