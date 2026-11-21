import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  PromptCandidateHistory,
  type CompilationHistorySummary,
} from "./prompt-candidate-history";

const fetchMock = vi.fn();
const history: CompilationHistorySummary = {
  available: true,
  candidate_count: 2,
  pending: false,
  configurations: [
    { index: 0, preset: "mipro_few_shot_v1", status: "completed", count: 2 },
  ],
};
const suggestion = {
  id: "unused",
  config_index: 0,
  preset: "mipro_few_shot_v1",
  proposal_order: 1,
  kind: "instruction",
  instructions_preview: "An unused instruction",
  demonstration_count: null,
  sources: ["proposal"],
  evaluations: [],
  selected: false,
  historical: false,
};
const selected = {
  ...suggestion,
  id: "chosen",
  proposal_order: 2,
  kind: "prompt",
  selected: true,
  instructions_preview: "Read the evidence carefully",
  demonstration_count: 1,
  evaluations: [
    {
      score: 0,
      sample_count: 2,
      partition: "training",
      scope: "minibatch",
      stage: "optimizer",
    },
    {
      score: 0.9,
      sample_count: 4,
      partition: "validation",
      scope: "full",
      stage: "final_validation",
    },
  ],
};
const detail = {
  ...selected,
  instructions:
    "Read the evidence carefully\nPreserve all citations. <script>literal text</script>",
  demonstrations: [
    {
      question: "Example question?",
      context: "[doc::1] Evidence",
      answer: "Supported answer [doc::1]",
    },
  ],
};
const page = { items: [suggestion, selected], total: 2, history };
const response = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status });
function mount(summary = history, status = "completed") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const result = render(
    <PromptCandidateHistory jobId="job" status={status} history={summary} />,
    {
      wrapper: ({ children }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    },
  );
  return { ...result, client };
}
function open() {
  fireEvent.click(screen.getByText(/View prompt suggestions/));
}
beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockReset();
  fetchMock.mockImplementation((url: string) =>
    Promise.resolve(response(url.includes("/candidates?") ? page : detail)),
  );
});

describe("DSPy prompt suggestion inspector", () => {
  it("loads lazily and shows unevaluated suggestions, zero scores, and complete selected prompts", async () => {
    mount();
    expect(fetchMock).not.toHaveBeenCalled();
    open();
    const table = await screen.findByRole("table", {
      name: "DSPy prompt suggestions",
    });
    expect(
      within(table).getByText("Not evaluated on full validation"),
    ).toBeInTheDocument();
    expect(within(table).getByText(/0\.000/)).toBeInTheDocument();
    expect(
      within(table).getByText(/training \/ minibatch/),
    ).toBeInTheDocument();
    expect(within(table).getByText("Selected by DSPy")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    fireEvent.click(
      screen.getByRole("button", {
        name: "View prompt 2 from configuration 1",
      }),
    );
    const content = await screen.findByRole("region", {
      name: "Prompt candidate details",
    });
    expect(
      (await within(content).findByText(/Preserve all citations/)).textContent,
    ).toBe(detail.instructions);
    expect(content.querySelector("script")).toBeNull();
    fireEvent.click(within(content).getByText("Examples (1)"));
    expect(within(content).getByText("[doc::1] Evidence")).toBeVisible();
    expect(
      within(content).getByText(/Final artifact validation: 0.900/),
    ).toBeInTheDocument();
    expect(
      within(content).getByText(
        /Answer-trial and holdout selection are separate/,
      ),
    ).toBeInTheDocument();
  });

  it("shows full validation ties separately from later training minibatch scores", async () => {
    const chosen = {
      ...selected,
      proposal_order: 1,
      evaluations: [
        {
          score: 1,
          sample_count: 6,
          partition: "validation",
          scope: "full",
          stage: "optimizer",
        },
        {
          score: 0.9,
          sample_count: 3,
          partition: "training",
          scope: "minibatch",
          stage: "optimizer",
        },
        {
          score: 1,
          sample_count: 6,
          partition: "validation",
          scope: "full",
          stage: "final_validation",
        },
      ],
    };
    const rival = {
      ...chosen,
      id: "rival",
      proposal_order: 2,
      selected: false,
      evaluations: [
        chosen.evaluations[0],
        { ...chosen.evaluations[1], score: 0.9667 },
      ],
    };
    fetchMock.mockResolvedValue(response({ ...page, items: [chosen, rival] }));
    mount();
    open();
    const table = await screen.findByRole("table", {
      name: "DSPy prompt suggestions",
    });
    expect(
      within(table).getByText("Full validation score"),
    ).toBeInTheDocument();
    const rows = within(table).getAllByRole("row").slice(1);
    expect(within(rows[0]).getAllByRole("cell")[5]).toHaveTextContent(
      "1.000Optimizer · 6 questions",
    );
    expect(within(rows[1]).getAllByRole("cell")[5]).toHaveTextContent(
      "1.000Optimizer · 6 questions",
    );
    expect(within(rows[0]).getAllByRole("cell")[6]).toHaveTextContent(
      "0.900training / minibatch · 3 questions",
    );
    expect(within(rows[1]).getAllByRole("cell")[6]).toHaveTextContent(
      "0.967training / minibatch · 3 questions",
    );
    expect(within(rows[0]).getAllByRole("cell")[7]).toHaveTextContent(
      "Selected by DSPy",
    );
  });

  it("supports server search, configuration filtering and pagination", async () => {
    fetchMock.mockResolvedValue(response({ ...page, total: 30 }));
    mount();
    open();
    await screen.findByRole("table");
    fireEvent.click(screen.getByRole("button", { name: "Next suggestions" }));
    await waitFor(() =>
      expect(fetchMock.mock.calls.at(-1)?.[0]).toContain("offset=25"),
    );
    fireEvent.change(screen.getByLabelText("Search prompt suggestions"), {
      target: { value: "evidence" },
    });
    await waitFor(() =>
      expect(fetchMock.mock.calls.at(-1)?.[0]).toContain("q=evidence"),
    );
    expect(fetchMock.mock.calls.at(-1)?.[0]).toContain("offset=0");
    fireEvent.change(screen.getByLabelText("Optimizer configuration"), {
      target: { value: "0" },
    });
    await waitFor(() =>
      expect(fetchMock.mock.calls.at(-1)?.[0]).toContain("config_index=0"),
    );
  });

  it("shows loading and retries list and detail errors", async () => {
    let finish!: (value: Response) => void;
    fetchMock.mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          finish = resolve;
        }),
    );
    mount();
    open();
    expect(
      await screen.findByText("Loading prompt suggestions..."),
    ).toBeInTheDocument();
    await act(async () =>
      finish(response({ detail: "History unavailable" }, 500)),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "History unavailable",
    );
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByRole("table");
    fetchMock.mockResolvedValueOnce(
      response({ detail: "Prompt unavailable" }, 500),
    );
    fireEvent.click(
      screen.getByRole("button", {
        name: "View prompt 2 from configuration 1",
      }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Prompt unavailable",
    );
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(
      (await screen.findByText(/Preserve all citations/)).textContent,
    ).toBe(detail.instructions);
  });

  it("refreshes completed history as the polled compilation summary changes", async () => {
    const pending = {
      available: false,
      candidate_count: 0,
      pending: true,
      configurations: [],
    };
    fetchMock.mockResolvedValueOnce(
      response({ items: [], total: 0, history: pending }),
    );
    const result = mount(pending, "running");
    open();
    await screen.findByText(/No completed suggestions match/);
    expect(screen.getByText(/Compilation is in progress/)).toBeInTheDocument();
    result.rerender(
      <PromptCandidateHistory jobId="job" status="running" history={history} />,
    );
    await screen.findByRole("table");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("explains historical output and handles unavailable scores", async () => {
    const legacy = {
      ...selected,
      historical: true,
      evaluations: [
        {
          score: null,
          sample_count: null,
          partition: "validation",
          scope: "full",
          stage: "final_validation",
        },
      ],
    };
    fetchMock.mockImplementation((url: string) =>
      Promise.resolve(
        response(
          url.includes("/candidates?")
            ? {
                items: [legacy],
                total: 1,
                history: { ...history, available: false, candidate_count: 0 },
              }
            : { ...detail, ...legacy },
        ),
      ),
    );
    mount();
    open();
    await screen.findByText("Candidate history was not recorded.");
    fireEvent.click(
      screen.getByRole("button", {
        name: "View prompt 2 from configuration 1",
      }),
    );
    await screen.findByText(/This is the saved selected prompt/);
    expect(
      screen.getByText(/Final artifact validation: Unavailable/),
    ).toBeInTheDocument();
  });

  it("explains BootstrapFewShot and standalone suggestions", async () => {
    fetchMock.mockImplementation((url: string) =>
      Promise.resolve(
        response(
          url.includes("/candidates?")
            ? {
                ...page,
                history: {
                  ...history,
                  configurations: [
                    {
                      index: 0,
                      preset: "bootstrap_small_v1",
                      status: "completed",
                    },
                  ],
                },
              }
            : {
                ...suggestion,
                instructions: "Unused instructions",
                demonstrations: null,
              },
        ),
      ),
    );
    mount();
    open();
    await screen.findByText(/BootstrapFewShot selects examples/);
    fireEvent.click(
      screen.getByRole("button", {
        name: "View prompt 1 from configuration 1",
      }),
    );
    await screen.findByText("Not evaluated as a standalone prompt.");
    expect(
      screen.getByText(/Examples are selected separately/),
    ).toBeInTheDocument();
  });
});
