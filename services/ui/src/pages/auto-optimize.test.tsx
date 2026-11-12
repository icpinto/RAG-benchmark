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

function overrideFetch(
  handler: (path: string) => ReturnType<typeof response> | undefined,
) {
  const fallback = fetchMock.getMockImplementation()!;
  fetchMock.mockImplementation(
    (path: string, options?: RequestInit) =>
      handler(path) ?? fallback(path, options),
  );
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
    if (path.endsWith("/trials") || path.endsWith("/events"))
      return response([]);
    if (path.endsWith("/recommendation")) return response(null);
    if (options?.method === "POST") return response({ status: "queued" });
    return response(session);
  });
});
describe("Auto Optimize", () => {
  it("submits bounded defaults and opens the captured session", async () => {
    mount();
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
    expect(payload.planner.provider).toBe("ollama");
    for (const role of ["planner", "synthesizer", "verifier", "judge"]) {
      expect(payload[role]).toMatchObject({
        model: "qwen2.5:32b",
        base_url: "http://100.110.250.66:11434",
      });
    }
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
    expect(run.queryByRole("textbox")).not.toBeInTheDocument();
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
    finished = true;
    act(() =>
      client.setQueryData(["optimization", "session"], {
        ...session,
        status: "completed",
        stage: "done",
      }),
    );
    await screen.findByText("No trial is currently running.");
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
