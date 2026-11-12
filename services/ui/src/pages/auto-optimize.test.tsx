import { render, screen, fireEvent, waitFor } from "@testing-library/react";
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
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
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
  });
});
