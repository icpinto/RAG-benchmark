import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { beforeEach, expect, it, vi } from "vitest";
import {
  PromptCompilationPanel,
  PromptOptimizerSettings,
  type PromptSettings,
} from "./prompt-optimization";

const fetchMock = vi.fn();
beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockReset();
});
const response = (payload: unknown) =>
  Promise.resolve(
    new Response(JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    }),
  );

it("enables the default preset and preserves selected compilation order", () => {
  let settings: PromptSettings = { configs: [] };
  function Harness() {
    const [value, setValue] = useState(settings);
    settings = value;
    return <PromptOptimizerSettings value={value} onChange={setValue} />;
  }
  render(<Harness />);
  fireEvent.click(screen.getByLabelText("Enable DSPy compilation"));
  expect(settings.configs[0].preset).toBe("mipro_zero_shot_v1");
  fireEvent.change(screen.getByLabelText("Auto Optimize answer shortlist"), {
    target: { value: "3" },
  });
  fireEvent.click(screen.getByLabelText("gepa_light_v1"));
  expect(settings.configs.map((c) => c.preset)).toEqual([
    "mipro_zero_shot_v1",
    "gepa_light_v1",
  ]);
  fireEvent.click(screen.getByRole("button", { name: "Move earlier" }));
  expect(settings.configs[0].preset).toBe("gepa_light_v1");
  expect(settings.answer_shortlist_size).toBe(3);
});

it("marks invalid advanced configuration JSON before compilation", () => {
  const valid = vi.fn();
  render(
    <PromptOptimizerSettings
      value={{ configs: [{ preset: "mipro_zero_shot_v1" }] }}
      onChange={vi.fn()}
      onValidityChange={valid}
    />,
  );
  fireEvent.change(
    screen.getByLabelText("Prompt optimizer configurations JSON"),
    { target: { value: "{" } },
  );
  expect(valid).toHaveBeenLastCalledWith(false);
});

it("compiles explicitly, inspects saved prompts and applies evaluation YAML", async () => {
  const apply = vi.fn();
  const prepare = vi.fn(async () => ({
    config_path: "manual.yaml",
    question_indices: [0, 1, 2],
  }));
  const settings = { configs: [{ preset: "mipro_zero_shot_v1" }] };
  const artifact = {
    id: "artifact",
    job_id: "job",
    summary: { preset: "mipro_zero_shot_v1", validation_score: 0.9 },
    payload: {
      instructions: "Use evidence",
      demonstrations: [],
      provenance: { answer_model: { model: "fixed" } },
    },
  };
  fetchMock.mockImplementation((input: string, options?: RequestInit) => {
    const path = String(input);
    if (path === "/api/prompt-artifacts") return response([artifact]);
    if (path.startsWith("/api/prompt-optimization/cases"))
      return response([{ id: "case1" }]);
    if (path === "/api/prompt-optimizations" && options?.method === "POST")
      return response({ id: "job" });
    if (path === "/api/prompt-optimizations/job")
      return response({
        id: "job",
        status: "completed",
        attempts: 1,
        progress: { results: [] },
        splits: {
          training: ["train"],
          validation: ["validation"],
          evaluation: ["eval"],
        },
        usage: { tokens: 50, reported_actual_tokens: 30, cost: null },
        artifacts: [artifact],
      });
    if (path === "/api/prompt-artifacts/artifact/config")
      return response({ yaml: "pipeline: compiled\n" });
    if (path === "/api/prompt-artifacts/artifact") return response(artifact);
    throw new Error(path);
  });
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <PromptCompilationPanel
        settings={settings}
        onSettingsChange={vi.fn()}
        prepare={prepare}
        onApply={apply}
      />
    </QueryClientProvider>,
  );
  expect(fetchMock).not.toHaveBeenCalled();
  const details = screen
    .getByText("Compile and reuse DSPy prompts")
    .closest("details")!;
  details.open = true;
  fireEvent(details, new Event("toggle"));
  fireEvent.click(
    await screen.findByRole("button", {
      name: "Compile selected configurations",
    }),
  );
  await screen.findByText(/Compilation completed:/);
  expect(prepare).toHaveBeenCalledOnce();
  const request = fetchMock.mock.calls.find(
    ([url, options]) =>
      url === "/api/prompt-optimizations" && options?.method === "POST",
  );
  expect(JSON.parse(request![1].body)).toMatchObject({
    case_id: "case1",
    prompt_optimization: settings,
    question_indices: [0, 1, 2],
  });
  fireEvent.change(screen.getByLabelText("Saved compiled prompt"), {
    target: { value: "artifact" },
  });
  await screen.findByText(/Validation correctness: 0.9/);
  fireEvent.click(
    screen.getByRole("button", {
      name: "Apply prompt and evaluation partition",
    }),
  );
  await waitFor(() =>
    expect(apply).toHaveBeenCalledWith("pipeline: compiled\n"),
  );
});
