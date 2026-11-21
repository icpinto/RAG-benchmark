import { expect, test } from "@playwright/test";
import { stageFixture } from "./stage-fixture";

for (const width of [1280, 375]) {
  test(`inspect DSPy suggestions and full examples at ${width}px`, async ({
    page,
  }) => {
    const history = {
      available: true,
      candidate_count: 2,
      pending: false,
      configurations: [
        {
          index: 0,
          preset: "mipro_few_shot_v1",
          status: "completed",
          count: 2,
        },
      ],
    };
    const suggestion = {
      id: "proposal",
      config_index: 0,
      preset: "mipro_few_shot_v1",
      proposal_order: 1,
      kind: "instruction",
      instructions_preview: "Unused instruction",
      demonstration_count: null,
      sources: ["proposal"],
      evaluations: [],
      selected: false,
      historical: false,
    };
    const chosen = {
      ...suggestion,
      id: "selected",
      proposal_order: 2,
      kind: "prompt",
      selected: true,
      instructions_preview: "Use supporting evidence",
      demonstration_count: 1,
      evaluations: [
        {
          score: 0,
          sample_count: 2,
          partition: "validation",
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
    const instructions = `Use supporting evidence.\nPreserve citations.\n${"LongInstruction".repeat(70)}`;
    let candidateRequests = 0;
    let detailRequests = 0;
    const writes: string[] = [];
    await page.route("**/api/**", async (route) => {
      const url = new URL(route.request().url());
      const path = url.pathname.replace("/api", "");
      if (route.request().method() !== "GET") writes.push(path);
      let data: unknown = null;
      if (path === "/health") data = { status: "ok" };
      else if (path === "/configs" || path === "/data/files")
        data = { files: [] };
      else if (path.endsWith("/trials") || path.endsWith("/events")) data = [];
      else if (path === "/optimizations/session")
        data = {
          id: "session",
          status: "completed",
          stage: "done",
          budget: { tokens: 100, active_seconds: 10, cost: null },
          progress: {},
        };
      else if (path.endsWith("/stages")) {
        const dashboard = stageFixture();
        dashboard.workflow_version = 3;
        dashboard.stages.answer.compilations = [
          {
            id: "prompt-job",
            status: "completed",
            progress: {},
            usage: { tokens: 100, cost: null },
            history,
          },
        ];
        data = dashboard;
      } else if (path.endsWith("/configuration"))
        data = {
          request: { config_path: "saved.yaml", limits: {} },
          template: { llm: { model: "fixed-answer" } },
        };
      else if (path === "/prompt-optimizations/prompt-job/candidates") {
        candidateRequests += 1;
        const q = url.searchParams.get("q");
        data = {
          items: q ? [chosen] : [suggestion, chosen],
          total: q ? 1 : 2,
          history,
        };
      } else if (
        path === "/prompt-optimizations/prompt-job/candidates/selected"
      ) {
        detailRequests += 1;
        data = {
          ...chosen,
          instructions,
          demonstrations: [
            {
              question: "Example question?",
              context: "[doc::1] Example evidence",
              answer: "Example answer [doc::1]",
            },
          ],
        };
      }
      await route.fulfill({ json: data });
    });
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/auto-optimize/session?stage=answer");
    await expect(
      page.getByRole("heading", { name: "DSPy prompt compilation" }),
    ).toBeVisible();
    expect(candidateRequests).toBe(0);
    await page.getByText(/View prompt suggestions/).click();
    const table = page.getByRole("table", { name: "DSPy prompt suggestions" });
    await expect(table.getByText("Not evaluated")).toBeVisible();
    await expect(table.getByText("0.000")).toBeVisible();
    expect(detailRequests).toBe(0);
    await page
      .getByRole("button", { name: "View prompt 2 from configuration 1" })
      .click();
    const content = page.getByRole("region", {
      name: "Prompt candidate details",
    });
    await expect(content.locator("pre").first()).toHaveText(instructions);
    await content.getByText("Examples (1)").click();
    await expect(content.getByText("[doc::1] Example evidence")).toBeVisible();
    await expect(
      content.getByText(/Final artifact validation: 0.900/),
    ).toBeVisible();
    await page.getByLabel("Search prompt suggestions").fill("supporting");
    await expect(table.getByText("Unused instruction")).toHaveCount(0);
    await expect(table.getByText("Use supporting evidence")).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    expect(writes).toEqual([]);
    await page.locator(".prompt-candidate-history").screenshot({
      path: `test-results/prompt-suggestions-${width}.png`,
    });
  });
}
