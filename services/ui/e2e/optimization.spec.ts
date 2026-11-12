import { test, expect } from "@playwright/test";

test("select corpus, review synthetic references, inspect recommendation and export", async ({
  page,
}) => {
  let resumed = false;
  let submitted: Record<string, unknown> = {};
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
    else if (path.endsWith("/trials")) payload = [];
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
