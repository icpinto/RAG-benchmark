import { test, expect } from "@playwright/test";
import { stageFixture } from "./stage-fixture";

for (const width of [1280, 375]) {
  test(`trace retrieval through answering and holdout at ${width}px`, async ({
    page,
  }) => {
    const requests: { path: string; method: string }[] = [];
    const trials = [
      {
        id: "retrieval-winning-config",
        phase: "retrieval" as const,
        status: "completed",
      },
      {
        id: "answer-winning-config",
        phase: "answer" as const,
        status: "completed",
      },
      {
        id: "holdout-winning-config",
        phase: "confirm" as const,
        status: "completed",
      },
    ];
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname.replace("/api", "");
      requests.push({ path, method: route.request().method() });
      let data: unknown = null;
      if (path === "/health") data = { status: "ok" };
      else if (path === "/configs") data = { files: [] };
      else if (path === "/data/files") data = { files: [] };
      else if (path === "/optimizations/session")
        data = {
          id: "session",
          status: "completed",
          stage: "done",
          budget: { tokens: 100, active_seconds: 10, cost: null },
          progress: {},
        };
      else if (path.endsWith("/stages")) data = stageFixture(trials);
      else if (path.endsWith("/trials")) data = trials;
      else if (path.endsWith("/events")) data = [];
      else if (path.endsWith("/configuration"))
        data = {
          request: { config_path: "saved.yaml", limits: {} },
          template: { llm: { model: "fixed-answer" } },
        };
      else if (path.endsWith("/config"))
        data = {
          config: {
            pipeline: {
              ingestion: {
                chunkers: [
                  { type: "recursive", chunk_size: 512, overlap: 0.1 },
                ],
              },
              query: {
                retrievers: [{ type: "dense" }],
                prompt_builders: [
                  {
                    type: "preset",
                    preset_id: "concise_v1",
                    template:
                      "Cite supporting evidence. Context: {context}. Question: {question}",
                  },
                ],
              },
            },
            llm: { model: "fixed-answer" },
            runtime: { mode: "end_to_end" },
          },
        };
      await route.fulfill({ json: data });
    });
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/auto-optimize/session?stage=retrieval");
    const retrieval = page.getByRole("table", {
      name: "retrieval-trials table",
    });
    await expect(
      retrieval.getByRole("link", { name: "retrieva" }),
    ).toBeVisible();
    await expect(retrieval.getByText("0.800").first()).toBeVisible();
    await expect(
      page.getByRole("combobox", { name: "Retrieval comparison metric" }),
    ).toBeHidden();
    await page
      .getByText("View score charts and comparisons", { exact: true })
      .focus();
    await page.keyboard.press("Enter");
    await page
      .getByRole("combobox", { name: "Retrieval comparison metric" })
      .selectOption("recall");
    await page
      .getByRole("combobox", { name: "Retrieval comparison k" })
      .selectOption("5");
    await page.getByRole("tab", { name: /Answering/ }).click();
    await expect(page).toHaveURL(/\?stage=answer$/);
    const answer = page.getByRole("table", { name: "answer-trials table" });
    await expect(answer.getByText("0.900").first()).toBeVisible();
    await answer.getByText("View details", { exact: true }).click();
    await expect(answer.getByText("Chunker · Inherited")).toBeVisible();
    await expect(
      answer.getByText("Prompt settings · Varied here"),
    ).toBeVisible();
    await answer
      .getByText("Full trial configuration YAML", { exact: true })
      .click();
    await expect(
      answer.getByLabel("Full trial configuration YAML"),
    ).toContainText("concise_v1");
    await page.screenshot({
      path: `test-results/stage-answer-${width}.png`,
      fullPage: true,
    });
    await answer.getByRole("link", { name: "retrieva" }).first().click();
    await expect(page).toHaveURL(
      /\?stage=retrieval#trial-retrieval-winning-config$/,
    );
    await expect(page.getByRole("tab", { name: /Retrieval/ })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await page.getByRole("tab", { name: /Confirmation/ }).click();
    await expect(
      page.getByText("confirmed improvement", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("img", { name: "95% confidence interval 0.200 to 0.600" }),
    ).toBeHidden();
    await page
      .getByText("View holdout chart and comparison", { exact: true })
      .click();
    await expect(
      page.getByRole("img", { name: "95% confidence interval 0.200 to 0.600" }),
    ).toBeVisible();
    await expect(
      page
        .getByRole("table", { name: "holdout-comparison table" })
        .getByText("0.9", { exact: true }),
    ).toBeVisible();
    await page.reload();
    await expect(
      page.getByRole("tab", { name: /Confirmation/ }),
    ).toHaveAttribute("aria-selected", "true");
    await expect(
      page.getByText("confirmed improvement", { exact: true }),
    ).toBeVisible();
    await page.screenshot({
      path: `test-results/stage-confirm-${width}.png`,
      fullPage: true,
    });
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
    ).toBeLessThanOrEqual(width);
    expect(requests.every((request) => request.method === "GET")).toBe(true);
  });
}
