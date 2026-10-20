import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import KnowledgeBase from "./knowledge-base";
import BenchmarkSetup, { selectionPayload } from "./benchmark-setup";
import { TraceExplorer } from "./results-explorer";
import { DraftProvider } from "../hooks/draft";
import catalog from "../test/catalog.json";
import { sampleConfig, sampleResults } from "../test/fixtures";
const fetchMock = vi.fn();
function mount(page: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const router = createMemoryRouter([
    { path: "/", element: <DraftProvider>{page}</DraftProvider> },
    { path: "/results/:runId", element: <div>Launched run</div> },
  ]);
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
}
function reply(data: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(data), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
}
beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockReset();
  fetchMock.mockImplementation((path: string) => {
    if (path === "/api/configs/schema") return reply(catalog);
    if (path === "/api/configs") return reply({ files: ["default.yaml"] });
    if (path.startsWith("/api/configs/content"))
      return reply({
        content: sampleConfig,
        config_path: "/app/configs/default.yaml",
      });
    if (path === "/api/data/files")
      return reply({
        files: [
          {
            name: "policy.md",
            path: "policy.md",
            size_bytes: 32,
            modified_at: 1,
          },
        ],
      });
    if (path === "/api/data/questions")
      return reply({
        questions: [
          {
            question: "Refund?",
            expected_answer: "14 days",
            ground_truth_doc_ids: [],
            ground_truth_chunk_ids: [],
            expected_citations: [],
          },
        ],
      });
    return reply({ status: "saved" });
  });
});
describe("workspace workflows", () => {
  it("uploads source documents using multipart file fields", async () => {
    mount(<KnowledgeBase />);
    const file = new File(["policy"], "policy.md", { type: "text/markdown" });
    fireEvent.change(screen.getByLabelText("Source documents"), {
      target: { files: [file] },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Upload source documents" }),
    );
    await screen.findByText("Source documents uploaded.");
    const call = fetchMock.mock.calls.find(
      ([path]) => path === "/api/data/upload",
    );
    expect(call?.[1].body.getAll("files")).toHaveLength(1);
    expect(call?.[1].headers["Content-Type"]).toBeUndefined();
  });
  it("requires confirmation before replacing ground truth", async () => {
    mount(<KnowledgeBase />);
    fireEvent.change(screen.getByLabelText("Ground-truth JSON"), {
      target: { files: [new File(["[]"], "qa.json")] },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Replace ground-truth JSON" }),
    );
    expect(
      fetchMock.mock.calls.some(([path]) => path === "/api/data/ground-truth"),
    ).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Replace questions" }));
    await screen.findByText("Ground-truth questions replaced.");
  });
  it("creates a question with all ground-truth fields", async () => {
    mount(<KnowledgeBase />);
    fireEvent.click(
      screen.getByRole("button", { name: "Evaluation questions" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Add question" }));
    fireEvent.change(screen.getByLabelText("Evaluation question"), {
      target: { value: "What is the policy?" },
    });
    fireEvent.change(screen.getByLabelText(/ground truth doc ids/), {
      target: { value: "a.md, b.md" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save question" }));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([path, options]) =>
            path === "/api/data/questions" && options?.method === "POST",
        ),
      ).toBe(true),
    );
    const call = fetchMock.mock.calls.find(
      ([path, options]) =>
        path === "/api/data/questions" && options?.method === "POST",
    );
    expect(JSON.parse(call?.[1].body).ground_truth_doc_ids).toEqual([
      "a.md",
      "b.md",
    ]);
  });
  it("does not queue a run after a failed save", async () => {
    const original = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((path: string, options?: RequestInit) =>
      options?.method === "PUT"
        ? reply({ detail: "Cannot save YAML" }, 400)
        : original(path, options),
    );
    mount(<BenchmarkSetup />);
    await screen.findByDisplayValue("research_run");
    fireEvent.click(screen.getByRole("button", { name: "Save and run" }));
    await screen.findAllByText("Cannot save YAML");
    expect(
      fetchMock.mock.calls.some(([path]) => path === "/api/benchmarks/run"),
    ).toBe(false);
  });
  it("saves the edited draft then queues the selected inputs", async () => {
    const original = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((path: string, options?: RequestInit) =>
      path === "/api/benchmarks/run"
        ? reply({ run_id: "new-run" })
        : original(path, options),
    );
    mount(<BenchmarkSetup />);
    const input = await screen.findByLabelText("Name");
    fireEvent.change(input, { target: { value: "edited_run" } });
    fireEvent.click(screen.getByRole("button", { name: "Save and run" }));
    await screen.findByText("Launched run");
    const save = fetchMock.mock.calls.find(
      ([, options]) => options?.method === "PUT",
    );
    expect(JSON.parse(save?.[1].body).content).toContain("edited_run");
    const call = fetchMock.mock.calls.find(
      ([path]) => path === "/api/benchmarks/run",
    );
    expect(JSON.parse(call?.[1].body)).toEqual({
      config_path: "/app/configs/default.yaml",
      source_paths: ["policy.md"],
      question_indices: [0],
    });
  });
  it("preserves omitted versus explicit-empty input selections", () => {
    expect(
      selectionPayload("a.yaml", "configured", [], "configured", []),
    ).toEqual({ config_path: "a.yaml" });
    expect(selectionPayload("a.yaml", "selected", [], "selected", [])).toEqual({
      config_path: "a.yaml",
      source_paths: [],
      question_indices: [],
    });
  });
  it("handles historical results without traces", () => {
    mount(
      <TraceExplorer
        results={[{ case_id: "old", question: "q", answer: "a" }]}
      />,
    );
    expect(screen.getByText(/historical result/)).toBeInTheDocument();
  });
  it("shows saved context, rewritten queries and chunk evidence", () => {
    mount(<TraceExplorer results={sampleResults} />);
    expect(
      screen.getByText("Refunds are accepted within 14 days."),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Prompt / query" }));
    expect(screen.getByText("refund window")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retrieved chunks" }));
    expect(screen.getByText(/policy.md::chunk-1/)).toBeInTheDocument();
  });
});
