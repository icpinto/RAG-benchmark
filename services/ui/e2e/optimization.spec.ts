import { test, expect } from "@playwright/test";
import { stageFixture } from "./stage-fixture";

for (const width of [1280, 375]) {
  test(`browse previous runs, reload details and start a new run at ${width}px`, async ({
    page,
  }) => {
    const runs = ["completed", "failed", "cancelled", "budget_exhausted"].map(
      (status, index) => ({
        id: `historical-optimization-${status}-00000000-1111-2222-3333-444444444444`,
        status,
        stage: "done",
        created_at: 1_790_000_000 - index * 3600,
        updated_at: 1_790_000_100 - index * 3600,
        budget: { tokens: 1000, active_seconds: 90, cost: null },
        progress: {},
      }),
    );
    const requested: string[] = [];
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname.replace("/api", "");
      requested.push(path);
      let payload: unknown = null;
      if (path === "/health") payload = { status: "ok" };
      else if (path === "/configs") payload = { files: ["default.yaml"] };
      else if (path === "/data/files") payload = { files: [] };
      else if (path === "/optimizations") payload = runs;
      else if (path.endsWith("/configuration"))
        payload = {
          request: { config_path: "historical-template.yaml", limits: {} },
          template: { llm: { model: "historical-answer" } },
        };
      else if (path.endsWith("/stages")) payload = stageFixture();
      else if (path.endsWith("/trials") || path.endsWith("/events"))
        payload = [];
      else if (path.endsWith("/dataset") || path.endsWith("/recommendation"))
        payload = null;
      else payload = runs.find((run) => path === `/optimizations/${run.id}`);
      await route.fulfill({ json: payload });
    });
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/auto-optimize");
    await expect(
      page.getByRole("heading", { name: "Run history" }),
    ).toBeVisible();
    await expect(page.getByText(/newest 100/)).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Start an optimization" }),
    ).toHaveCount(0);
    for (const run of runs) {
      await page
        .getByRole("link", {
          name: `View details for run ${run.id}`,
          exact: true,
        })
        .click();
      await expect(page).toHaveURL(new RegExp(`/auto-optimize/${run.id}$`));
      await expect(
        page.getByRole("heading", { name: `Optimization · ${run.status}` }),
      ).toBeVisible();
      await expect(page.getByText(run.id, { exact: true })).toBeVisible();
      await expect(
        page.getByText("historical-template.yaml", { exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("button", { name: "Cancel", exact: true }),
      ).toHaveCount(0);
      await page.reload();
      await expect(
        page.getByRole("heading", { name: `Optimization · ${run.status}` }),
      ).toBeVisible();
      await expect(
        page.getByText("historical-template.yaml", { exact: true }),
      ).toBeVisible();
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(width);
      await page.getByRole("link", { name: "Back to run history" }).click();
      await expect(
        page.getByRole("table", { name: "optimization runs table" }),
      ).toBeVisible();
    }
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
    ).toBeLessThanOrEqual(width);
    if (width === 375) {
      const scrolling = await page.getByRole("table").evaluate((table) => {
        const container = table.parentElement!;
        return {
          scroll: container.scrollWidth,
          client: container.clientWidth,
          overflow: getComputedStyle(container).overflowX,
        };
      });
      expect(scrolling.scroll).toBeGreaterThan(scrolling.client);
      expect(scrolling.overflow).toBe("auto");
      await page.screenshot({
        path: "test-results/optimization-history-mobile.png",
        fullPage: true,
      });
    }
    await page
      .getByRole("link", { name: "New optimization", exact: true })
      .click();
    await expect(page).toHaveURL(/\/auto-optimize\/new$/);
    await expect(
      page.getByRole("heading", { name: "Start an optimization" }),
    ).toBeVisible();
    expect(requested).not.toContain("/optimizations/new");
    await page.getByRole("link", { name: "Back to run history" }).click();
    await expect(
      page.getByRole("heading", { name: "Run history" }),
    ).toBeVisible();
  });
}

test("select corpus, review synthetic references, inspect recommendation and export", async ({
  page,
}) => {
  let resumed = false;
  let submitted: Record<string, unknown> = {};
  const trialConfig = {
    pipeline: {
      ingestion: {
        chunkers: [{ type: "recursive", chunk_size: 512, overlap: 0.1 }],
        embeddings: [
          { provider: "sentence_transformers", model: "fixture-embedding" },
        ],
      },
      query: { retrievers: [{ type: "dense" }] },
    },
    runtime: { mode: "end_to_end" },
    metrics: { enabled: ["answer_correctness"] },
  };
  const question = {
    question_id: "q1",
    question: "What is the refund window?",
    expected_answer: "Fourteen days.",
    category: "fact",
    split: "tuning",
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
  };
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace("/api", "");
    const session = {
      id: "fixture",
      status: resumed ? "completed" : "paused",
      stage: resumed ? "done" : "freeze",
      budget: { tokens: 1000, active_seconds: 30, cost: null },
      progress: {},
    };
    let payload: unknown = {};
    if (path === "/health") payload = { status: "ok" };
    else if (path === "/configs") payload = { files: ["default.yaml"] };
    else if (path === "/data/files")
      payload = {
        files: [{ name: "policy.txt", path: "policy.txt", size_bytes: 100 }],
      };
    else if (path === "/optimizations" && route.request().method() === "POST") {
      submitted = route.request().postDataJSON();
      payload = { id: "fixture" };
    } else if (path === "/optimizations") payload = [session];
    else if (path.endsWith("/resume")) {
      resumed = true;
      payload = { status: "queued" };
    } else if (path.endsWith("/dataset"))
      payload = { version: 1, frozen: resumed ? 1 : 0, payload: [question] };
    else if (path.endsWith("/configuration"))
      payload = {
        request: { ...submitted, config_path: "captured-template.yaml" },
        template: { llm: { provider: "ollama", model: "fixed-answer" } },
      };
    else if (path.endsWith("/trials"))
      payload = [
        {
          id: "fixture-trial",
          phase: "answer",
          split: "tuning",
          status: resumed ? "completed" : "paused",
        },
      ];
    else if (path.endsWith("/config")) payload = { config: trialConfig };
    else if (path.endsWith("/stages"))
      payload = stageFixture([
        { id: "saved-trial", phase: "answer", status: "completed" },
      ]);
    else if (path.endsWith("/events")) payload = [];
    else if (path.endsWith("/recommendation"))
      payload = resumed
        ? {
            yaml: "experiment:\n  name: optimized\n",
            confirmed_improvement: false,
            baseline_retained: true,
            limitations: ["Synthetic reference data is not human gold"],
          }
        : null;
    else if (path.endsWith("/export/yaml")) {
      await route.fulfill({
        contentType: "application/yaml",
        headers: {
          "Content-Disposition": 'attachment; filename="recommended.yaml"',
        },
        body: "experiment:\n  name: optimized\n",
      });
      return;
    } else payload = session;
    await route.fulfill({ json: payload });
  });
  await page.goto("/auto-optimize");
  await page
    .getByRole("link", { name: "New optimization", exact: true })
    .click();
  await page.getByLabel("Source documents").selectOption("selected");
  await page.getByRole("checkbox", { name: "policy.txt" }).check();
  await page
    .getByRole("checkbox", {
      name: "Pause for dataset review before experiments",
    })
    .check();
  await page.getByRole("button", { name: "Start optimization" }).click();
  await expect(page.getByText("Optimization · paused")).toBeVisible();
  expect(submitted.source_paths).toEqual(["policy.txt"]);
  await expect(
    page.getByText("captured-template.yaml", { exact: true }),
  ).toBeVisible();
  await page.getByText("Saved request YAML", { exact: true }).click();
  await expect(page.getByLabel("Saved request YAML")).toContainText(
    "target_questions: 100",
  );
  await expect(page.getByText("No trial is currently running.")).toBeVisible();
  await page.getByRole("tab", { name: /Answering/ }).click();
  await page.getByText("View config", { exact: true }).click();
  await expect(
    page
      .locator("dd")
      .filter({ hasText: /^type: recursive, chunk_size: 512, overlap: 0\.1$/ }),
  ).toBeVisible();
  await page
    .getByText("Full trial configuration YAML", { exact: true })
    .click();
  await expect(page.getByLabel("Full trial configuration YAML")).toContainText(
    "answer_correctness",
  );
  await page.reload();
  await expect(
    page.getByText("captured-template.yaml", { exact: true }),
  ).toBeVisible();
  await page.getByText("Inspect questions and source evidence").click();
  await expect(
    page.getByText("Refunds within fourteen days.", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Approve draft and resume" }).click();
  await expect(page.getByText("Optimization · completed")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Open in Benchmark Setup" }),
  ).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download YAML" }).click();
  expect((await download).suggestedFilename()).toBe("recommended.yaml");
});

test("inspect running configs on a narrow screen and follow trial completion", async ({
  page,
}) => {
  let finished = false;
  const longModel = `local-model-${"long-name-".repeat(30)}`;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname.replace("/api", "");
    const session = {
      id: "live",
      status: finished ? "completed" : "running",
      stage: finished ? "done" : "retrieval",
      budget: { tokens: 1000, active_seconds: 30, cost: null },
      progress: {},
    };
    let payload: unknown = null;
    if (path === "/health") payload = { status: "ok" };
    else if (path === "/configs") payload = { files: ["default.yaml"] };
    else if (path === "/data/files") payload = { files: [] };
    else if (path === "/optimizations") payload = [session];
    else if (path === "/optimizations/live") payload = session;
    else if (path.endsWith("/configuration"))
      payload = {
        request: {
          config_path: "saved.yaml",
          planner: { model: longModel },
          limits: {},
          allowed_components: {},
        },
        template: { llm: { model: "fixed-answer" } },
      };
    else if (path.endsWith("/trials"))
      payload = [
        {
          id: "live-trial",
          phase: "retrieval",
          split: "tuning",
          status: finished ? "completed" : "running",
        },
      ];
    else if (path.endsWith("/config"))
      payload = {
        config: {
          pipeline: {
            ingestion: {
              chunkers: [{ type: "fixed", chunk_size: 256 }],
              embeddings: [{ model: longModel }],
            },
          },
          runtime: { mode: "retrieval_only" },
        },
      };
    else if (path.endsWith("/stages"))
      payload = stageFixture([
        {
          id: "live-trial",
          phase: "retrieval",
          status: finished ? "completed" : "running",
        },
      ]);
    else if (path.endsWith("/events")) payload = [];
    await route.fulfill({ json: payload });
  });
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("/auto-optimize/live");
  const running = page.getByRole("article", {
    name: "Running trial live-trial",
  });
  await expect(
    running.getByText("type: fixed, chunk_size: 256", { exact: true }),
  ).toBeVisible();
  await running
    .getByText("Full trial configuration YAML", { exact: true })
    .click();
  await expect(
    running.getByLabel("Full trial configuration YAML"),
  ).toContainText(longModel);
  const widths = await running
    .getByLabel("Full trial configuration YAML")
    .evaluate((element) => ({
      scroll: element.scrollWidth,
      client: element.clientWidth,
      overflow: getComputedStyle(element).overflowX,
    }));
  expect(widths.scroll).toBeGreaterThan(widths.client);
  expect(widths.overflow).toBe("auto");
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(375);
  await page.screenshot({
    path: "test-results/optimization-config-mobile.png",
    fullPage: true,
  });
  finished = true;
  await expect(page.getByText("Optimization · completed")).toBeVisible();
  await expect(page.getByText("No trial is currently running.")).toBeVisible();
  await page.getByRole("tab", { name: /Retrieval/ }).click();
  await page.getByText("View config", { exact: true }).click();
  await expect(page.getByText("retrieval_only", { exact: true })).toBeVisible();
});
