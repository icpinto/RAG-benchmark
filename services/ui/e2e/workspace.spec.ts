import { expect, test, type Page } from "@playwright/test";
import catalog from "../src/test/catalog.json" with { type: "json" };
import { sampleConfig, sampleResults, sampleRuns } from "../src/test/fixtures";
import { expectReadableLayout } from "./readability";
async function backend(page: Page) {
  let config = sampleConfig;
  let failSave = false;
  let failFiles = false;
  let runReads = 0;
  const launched: unknown[] = [];
  const uploads: string[] = [];
  let questions = [
    {
      question: "What is the refund window?",
      expected_answer: "14 days.",
      ground_truth_doc_ids: ["policy.md"],
      ground_truth_chunk_ids: [],
      expected_citations: ["policy.md"],
    },
  ];
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname.replace(/^\/api/, "");
    const method = request.method();
    const respond = (value: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(value),
      });
    if (path === "/health") return respond({ status: "ok" });
    if (path === "/configs/schema") return respond(catalog);
    if (path === "/configs")
      return respond({ files: ["default.yaml"], config_root: "/app/configs" });
    if (path === "/configs/content") {
      if (method === "PUT") {
        if (failSave)
          return respond({ detail: "Configuration is read-only" }, 400);
        config = request.postDataJSON().content;
      }
      return respond({
        content: config,
        config_path: "/app/configs/default.yaml",
        status: "saved",
      });
    }
    if (path === "/data/files")
      return failFiles
        ? respond({ detail: "Document service unavailable" }, 503)
        : respond({
            data_root: "/app/data/raw",
            files: [
              {
                name: "policy.md",
                path: "policy.md",
                size_bytes: 100,
                modified_at: 1,
              },
            ],
          });
    if (path === "/data/files/content")
      return respond({
        content: "Refunds are accepted within 14 days.",
        path: "policy.md",
      });
    if (path === "/data/upload") {
      uploads.push(request.postData() || "");
      return respond({ saved: ["policy.md"] });
    }
    if (path === "/data/ground-truth")
      return respond({ saved: "/app/data/eval/qa.json" });
    if (path === "/data/questions") {
      if (method === "POST") questions.push(request.postDataJSON());
      return respond({ path: "/app/data/eval/qa.json", questions });
    }
    if (/^\/data\/questions\/\d+$/.test(path)) {
      const index = Number(path.split("/").at(-1));
      if (method === "DELETE")
        questions = questions.filter((_, i) => i !== index);
      else questions[index] = request.postDataJSON();
      return respond({ status: "updated" });
    }
    if (path === "/benchmarks/run") {
      launched.push(request.postDataJSON());
      return respond({ run_id: "run-new", status: "queued" });
    }
    if (path === "/benchmarks") return respond(sampleRuns);
    if (path === "/benchmarks/compare")
      return respond({
        runs: sampleRuns,
        results: {
          "run-one": sampleResults,
          "run-two": sampleResults.map((row) => ({
            ...row,
            metrics: { "recall@5": 1 },
          })),
        },
        deltas: [],
      });
    if (path === "/benchmarks/run-new") {
      runReads++;
      return respond({
        ...sampleRuns[0],
        id: "run-new",
        status:
          runReads === 1 ? "queued" : runReads === 2 ? "running" : "completed",
      });
    }
    if (path.endsWith("/results")) return respond(sampleResults);
    const run = sampleRuns.find((run) => path === `/benchmarks/${run.id}`);
    if (run) return respond(run);
    return respond({ detail: "Not found" }, 404);
  });
  return {
    launched,
    uploads,
    failSave: () => {
      failSave = true;
    },
    failFiles: (value: boolean) => {
      failFiles = value;
    },
  };
}
test("data preparation, guided editing, launch, polling, traces and comparisons", async ({
  page,
}) => {
  const state = await backend(page);
  await page.goto("/knowledge-base");
  await expect(
    page.getByRole("heading", { name: "Knowledge Base", exact: true }),
  ).toBeVisible();
  await page.getByLabel("Source documents", { exact: true }).setInputFiles({
    name: "policy.md",
    mimeType: "text/markdown",
    buffer: Buffer.from("policy"),
  });
  await page.screenshot({
    path: "test-results/knowledge-base-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Upload source documents" }).click();
  await expect(page.getByText("Source documents uploaded.")).toBeVisible();
  expect(state.uploads[0]).toContain("policy.md");
  await page.getByLabel("Preview source document").selectOption("policy.md");
  await expect(
    page.getByText("Refunds are accepted within 14 days."),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Evaluation questions", exact: true })
    .click();
  await page.getByRole("button", { name: "Add question" }).click();
  await page
    .getByLabel("Evaluation question", { exact: true })
    .fill("What is excluded?");
  await page.getByRole("button", { name: "Save question" }).click();
  await expect(page.getByText("Question saved.")).toBeVisible();
  await page.getByRole("link", { name: /Benchmark Setup/ }).click();
  await page.getByLabel("Name", { exact: true }).fill("react_experiment");
  await expect(
    page.getByText("Unsaved changes", { exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/setup-guided-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "YAML source", exact: true }).click();
  await expect(page.locator(".cm-content")).toContainText("react_experiment");
  await page.screenshot({
    path: "test-results/setup-desktop.png",
    fullPage: true,
  });
  await expect(page.locator(".cm-content")).toContainText("react_experiment");
  await page.getByRole("button", { name: "Save and run" }).click();
  await expect(page).toHaveURL(/results\/run-new/);
  await expect(
    page.getByText("This run is queued.", { exact: false }),
  ).toBeVisible();
  await expect(page.locator(".selected-run .status-badge")).toHaveText(
    "completed",
    { timeout: 15000 },
  );
  expect(state.launched[0]).toEqual({
    config_path: "/app/configs/default.yaml",
    source_paths: ["policy.md"],
    question_indices: [0, 1],
  });
  await page.getByRole("button", { name: "Scored results & traces" }).click();
  await expect(
    page.getByRole("heading", { name: "Trace Explorer" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Retrieved chunks" }).click();
  await expect(page.getByText(/policy.md::chunk-1/)).toBeVisible();
  await page.getByRole("button", { name: "Retrieval heatmap" }).click();
  await expect(page.getByText("0.500", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Compare runs" }).click();
  await page.getByRole("checkbox").nth(0).check();
  await page.getByRole("checkbox").nth(1).check();
  await expect(
    page.getByRole("heading", { name: "Compare benchmark runs" }),
  ).toBeVisible();
  await expect(page.getByText("0.5", { exact: true }).first()).toBeVisible();
  await page.screenshot({
    path: "test-results/comparison-desktop.png",
    fullPage: true,
  });
});
test("question updates and deletes use the current index and require confirmation", async ({
  page,
}) => {
  await backend(page);
  await page.goto("/knowledge-base");
  await page
    .getByRole("button", { name: "Evaluation questions", exact: true })
    .click();
  await page.getByRole("button", { name: "Edit 1", exact: true }).click();
  await page.getByLabel("Expected reference answer").fill("Updated answer");
  await page.getByRole("button", { name: "Save question" }).click();
  await expect(page.getByText("Updated answer", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByRole("alertdialog")).toBeVisible();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByText("Updated answer", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await page
    .getByRole("button", { name: "Delete question", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Define your first question" }),
  ).toBeVisible();
});
test("save errors do not queue runs and drafts are guarded", async ({
  page,
}) => {
  const state = await backend(page);
  state.failSave();
  await page.goto("/benchmark-setup");
  await page.getByLabel("Name", { exact: true }).fill("unsaved");
  await page.getByRole("button", { name: "Save and run" }).click();
  await expect(
    page.getByText("Configuration is read-only").first(),
  ).toBeVisible();
  expect(state.launched).toHaveLength(0);
  await page.getByRole("link", { name: /Knowledge Base/ }).click();
  await expect(page.getByRole("alertdialog")).toBeVisible();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByLabel("Name", { exact: true })).toHaveValue("unsaved");
  await page.getByRole("link", { name: /Knowledge Base/ }).click();
  await page.getByRole("button", { name: "Discard and leave" }).click();
  await expect(page).toHaveURL(/knowledge-base/);
});
test("mobile navigation, replacement confirmation, recovery and direct-link refresh", async ({
  page,
}) => {
  const state = await backend(page);
  await page.setViewportSize({ width: 390, height: 844 });
  state.failFiles(true);
  await page.goto("/knowledge-base");
  await expect(page.getByText("Document service unavailable")).toBeVisible();
  state.failFiles(false);
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(
    page.getByText("policy.md", { exact: true }).first(),
  ).toBeVisible();
  await page.getByLabel("Ground-truth JSON", { exact: true }).setInputFiles({
    name: "qa.json",
    mimeType: "application/json",
    buffer: Buffer.from("[]"),
  });
  await page.getByRole("button", { name: "Replace ground-truth JSON" }).click();
  await expect(page.getByRole("alertdialog")).toBeVisible();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("link", { name: /Results Explorer/ }).click();
  await expect(page).toHaveURL(/results/);
  await page.goto("/results/run-one?view=scores");
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Trace Explorer" }),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/results-mobile.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
});

for (const width of [390, 768, 1280, 1440]) {
  test(`readable workspaces and long content at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1000 });
    await backend(page);
    const filename = `research-${"very-long-document-name-".repeat(12)}.md`;
    const question =
      "How does the policy apply when the customer requests a refund? ".repeat(
        4,
      );
    const answer =
      "Refunds are accepted within 14 days. ".repeat(12) +
      "\n\nThe answer keeps its paragraphs and literal <strong>text</strong>.";
    await page.route("**/api/data/files", (route) =>
      route.fulfill({
        json: {
          data_root: "/app/data/raw",
          files: [
            {
              name: filename,
              path: filename,
              size_bytes: 100,
              modified_at: 1,
            },
          ],
        },
      }),
    );
    await page.route("**/api/data/questions", (route) =>
      route.fulfill({
        json: {
          path: "/app/data/eval/qa.json",
          questions: [
            {
              question,
              expected_answer: answer,
              ground_truth_doc_ids: [filename],
              ground_truth_chunk_ids: [],
              expected_citations: [],
            },
          ],
        },
      }),
    );
    await page.route("**/api/benchmarks/run-one/results", (route) =>
      route.fulfill({ json: [{ ...sampleResults[0], question, answer }] }),
    );
    await page.goto("/knowledge-base");
    await expect(
      page.getByRole("table", { name: "documents table" }),
    ).toBeVisible();
    await expectReadableLayout(page);
    await page.screenshot({
      path: `test-results/readability-${width}-knowledge-base.png`,
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Evaluation questions", exact: true })
      .click();
    const questionCell = page
      .getByRole("table", { name: "questions table" })
      .getByText(question, { exact: true });
    await expect(questionCell).toHaveCSS("white-space", "pre-wrap");
    expect(
      await questionCell.evaluate((el) => el.clientHeight),
    ).toBeGreaterThan(40);
    await expectReadableLayout(page);
    await page
      .getByRole("button", { name: "Add question", exact: true })
      .click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expectReadableLayout(page);
    await page.getByRole("button", { name: "Close dialog" }).click();

    await page.goto("/benchmark-setup");
    await expect(page.getByLabel("Name", { exact: true })).toBeVisible();
    await page.getByLabel("Name", { exact: true }).focus();
    await expect(page.getByLabel("Name", { exact: true })).toHaveCSS(
      "outline-style",
      "solid",
    );
    await expectReadableLayout(page);
    if (width < 1200) {
      await expect(
        page.getByLabel("Configuration section", { exact: true }),
      ).toBeVisible();
      await page
        .getByLabel("Configuration section", { exact: true })
        .selectOption("runtime");
    } else {
      await page.getByRole("button", { name: "Runtime", exact: true }).click();
    }
    await expect(page.locator(".config-content h2")).toHaveText("Runtime");
    await page
      .locator(".nested-fields summary")
      .filter({ hasText: "Embedding cache" })
      .click();
    await expect(page.locator(".nested-fields[open]")).toBeVisible();
    await expectReadableLayout(page);
    await page.screenshot({
      path: `test-results/readability-${width}-setup.png`,
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "YAML source", exact: true })
      .click();
    await expect(page.locator(".cm-content")).toContainText("research_run");
    await expectReadableLayout(page);

    await page.goto("/results/run-one?view=scores");
    await expect(
      page.getByRole("heading", { name: "Trace Explorer" }),
    ).toBeVisible();
    await page
      .getByText("Question and generated answer", { exact: true })
      .click();
    await expect(page.locator(".answer-text")).toHaveText(answer);
    await expect(page.locator(".answer-text strong")).toHaveCount(0);
    await expect(page.locator(".answer-text")).toHaveCSS(
      "white-space",
      "pre-wrap",
    );
    await expectReadableLayout(page);
    await page.screenshot({
      path: `test-results/readability-${width}-results.png`,
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Retrieval heatmap", exact: true })
      .click();
    await expect(page.locator(".heatmap td")).toHaveText("0.500");
    await expectReadableLayout(page);
  });
}

test.describe("200 percent browser zoom equivalent", () => {
  // A 1440x1000 display at 200% zoom has a 720x500 CSS viewport. Keep
  // media queries and device pixels consistent with that browser behavior.
  test.use({
    viewport: { width: 720, height: 500 },
    deviceScaleFactor: 2,
  });
  test("all workspaces reflow without clipped content", async ({ page }) => {
    await backend(page);
    for (const [name, path] of [
      ["knowledge-base", "/knowledge-base"],
      ["setup", "/benchmark-setup"],
      ["results", "/results/run-one?view=scores"],
    ]) {
      await page.goto(path);
      await expect(page.locator("main h1")).toBeVisible();
      await expectReadableLayout(page);
      await page.screenshot({
        path: `test-results/zoom-200-${name}.png`,
        fullPage: true,
      });
    }
  });
});
