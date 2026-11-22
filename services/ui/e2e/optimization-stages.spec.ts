import { test, expect } from "@playwright/test";
import { stageFixture } from "./stage-fixture";
import { expectReadableLayout } from "./readability";

for (const width of [1440, 1280, 375]) {
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
      retrieval.getByRole("columnheader", { name: /reranker/i }),
    ).toHaveCount(0);
    await expect(page.getByText(/Reranking is disabled/)).toBeVisible();
    await expect(
      page.getByRole("table", { name: "raw-and-reranked table" }),
    ).toHaveCount(0);
    await page.getByRole("tab", { name: /Answering/ }).click();
    await expect(page).toHaveURL(/\?stage=answer$/);
    const answer = page.getByRole("table", { name: "answer-trials table" });
    await expect(answer.getByText("0.900").first()).toBeVisible();
    await answer.getByText("View config", { exact: true }).click();
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
    await expectReadableLayout(page);
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

  test(`configuration review, errors and empty states at ${width}px`, async ({
    page,
  }) => {
    let review = false;
    let configurationFailed = false;
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname.replace("/api", "");
      if (configurationFailed && path.endsWith("/configuration")) {
        await route.fulfill({
          status: 503,
          json: { detail: "Saved settings temporarily unavailable" },
        });
        return;
      }
      let data: unknown = null;
      if (path === "/health") data = { status: "ok" };
      else if (path === "/configs" || path === "/data/files")
        data = { files: [] };
      else if (path === "/optimizations/session")
        data = {
          id: "session",
          status: review ? "needs_input" : "running",
          stage: "freeze",
          budget: { tokens: 25000, active_seconds: 60, cost: null },
          progress: {},
        };
      else if (path.endsWith("/configuration"))
        data = {
          request: {
            config_path: "review-template.yaml",
            limits: { target_questions: 30 },
          },
          template: { llm: { model: "fixed-answer" } },
        };
      else if (path.endsWith("/dataset"))
        data = {
          version: 1,
          frozen: review ? 0 : 1,
          evaluation_dataset: {
            status: "generated",
            dataset_id: "qa",
            source_dataset_id: "qa",
            source_session_id: null,
            question_count: 1,
          },
          payload: [
            {
              question_id: "q1",
              question: "What is the refund window?",
              expected_answer: "Fourteen days.",
              category: "fact",
              split: "tuning",
              automated_validated: true,
              human_reviewed: false,
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
      else if (path.endsWith("/stages")) data = stageFixture();
      else if (path.endsWith("/trials") || path.endsWith("/events")) data = [];
      await route.fulfill({ json: data });
    });
    await page.setViewportSize({ width, height: 1000 });
    await page.goto("/auto-optimize/session");
    const configuration = page.locator(".run-configuration-disclosure");
    await expect(configuration.locator("summary").first()).toContainText(
      "review-template.yaml",
    );
    await expect(configuration).not.toHaveAttribute("open");
    await expect(page.getByText("No trials in this stage yet")).toBeVisible();
    await expectReadableLayout(page);
    await page.screenshot({
      path: `test-results/optimizer-collapsed-${width}.png`,
      fullPage: true,
    });
    await configuration.locator("summary").first().focus();
    await page.keyboard.press("Enter");
    await expect(configuration).toHaveAttribute("open", "");
    await configuration.locator("summary").first().click();
    review = true;
    await expect(
      page.getByText("Review required", { exact: true }),
    ).toBeVisible({ timeout: 10000 });
    await expect(configuration).toHaveAttribute("open", "");
    await expect(
      page.getByRole("heading", { name: "Evaluation dataset", exact: true }),
    ).toBeVisible();
    await page
      .getByText("Inspect questions and source evidence", { exact: true })
      .click();
    await expect(
      page.getByText("Refunds within fourteen days.", { exact: true }),
    ).toBeVisible();
    await page.getByText("Edit draft questions", { exact: true }).click();
    await expect(
      page.getByLabel("Draft evaluation questions JSON"),
    ).toBeVisible();
    await expectReadableLayout(page);
    await page.screenshot({
      path: `test-results/optimizer-review-${width}.png`,
      fullPage: true,
    });
    await configuration.locator("summary").first().click();
    await expect(configuration).not.toHaveAttribute("open");
    // Observe another dataset poll: an unchanged review requirement must not reopen it.
    await page.waitForResponse((response) =>
      response.url().endsWith("/dataset"),
    );
    await expect(configuration).not.toHaveAttribute("open");
    configurationFailed = true;
    await page.reload();
    await expect(
      page.getByText("Saved settings temporarily unavailable"),
    ).toBeVisible();
    await expect(
      page.getByText("Edit draft questions", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Approve draft and resume" }),
    ).toBeEnabled();
    await expectReadableLayout(page);
    await page.screenshot({
      path: `test-results/optimizer-configuration-error-${width}.png`,
      fullPage: true,
    });
  });
}
