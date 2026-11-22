import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { stageFixture } from "./stage-fixture";
import type { AcceptanceDecision } from "../src/components/recommendation-evidence";

for (const [outcome, label] of [
  ["confirmed_improvement", "Confirmed improvement"],
  ["promising_candidate", "Promising candidate"],
  ["baseline_retained", "Baseline retained"],
] as const) {
  test(`${label} stays consistent through dashboard, download and Benchmark Setup`, async ({
    page,
  }) => {
    const baseline = outcome === "baseline_retained";
    const source = baseline ? "baseline" : "candidate";
    const yaml = `experiment:\n  name: recommended-${source}\n`;
    const explanation = {
      text: `The ${source} is recommended. These results use synthetic evaluation data.`,
      source: "llm" as const,
      prompt_version: "recommendation-explanation-v1",
    };
    const acceptance: AcceptanceDecision = {
      outcome,
      recommended_source: source,
      reason_codes: [],
      checks: [
        {
          code: "paired_coverage",
          label: "At least 95% paired scores",
          status: "pass",
          measured: { paired: 6, expected: 6, fraction: 1 },
          threshold: 0.95,
          required: true,
        },
      ],
    };
    const stages = stageFixture();
    Object.assign(stages.stages.confirm, {
      acceptance,
      explanation,
      outcome: outcome.replaceAll("_", " "),
      baseline_retained: baseline,
    });
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname.replace("/api", "");
      let data: unknown = null;
      if (path === "/health") data = { status: "ok" };
      else if (path === "/configs") data = { files: [] };
      else if (path === "/configs/schema") data = { stages: [], sections: [] };
      else if (path === "/data/files") data = { files: [] };
      else if (path === "/data/questions") data = { questions: [] };
      else if (path.endsWith("/stages")) data = stages;
      else if (path.endsWith("/recommendation"))
        data = {
          yaml,
          acceptance,
          explanation,
          confirmed_improvement: outcome === "confirmed_improvement",
          baseline_retained: baseline,
          limitations: [],
        };
      else if (path.endsWith("/configuration"))
        data = {
          request: { config_path: "saved.yaml", limits: {} },
          template: {},
        };
      else if (
        path.endsWith("/events") ||
        path.endsWith("/trials") ||
        path === "/prompt-artifacts"
      )
        data = [];
      else if (path === "/optimizations/session")
        data = {
          id: "session",
          status: "completed",
          stage: "done",
          progress: {},
          budget: { tokens: 100, active_seconds: 1, cost: null },
        };
      await route.fulfill({ json: data });
    });
    await page.goto("/auto-optimize/session?stage=confirm");
    await expect(page.getByText(label, { exact: true })).toHaveCount(2);
    await expect(page.getByText(explanation.text, { exact: true })).toHaveCount(
      2,
    );
    await expect(
      page.getByText(
        baseline
          ? "Baseline recommendation retained."
          : "Candidate recommended.",
        { exact: false },
      ),
    ).toBeVisible();
    const downloaded = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download YAML" }).click();
    const download = await downloaded;
    expect(download.suggestedFilename()).toBe("recommended.yaml");
    expect(await readFile((await download.path())!, "utf8")).toBe(yaml);
    await page.getByRole("button", { name: "Open in Benchmark Setup" }).click();
    await expect(page).toHaveURL(/\/benchmark-setup$/);
      await page.getByRole("button", { name: "YAML source" }).click();
    await expect(page.locator(".cm-content")).toContainText(
      `recommended-${source}`,
    );
  });
}
