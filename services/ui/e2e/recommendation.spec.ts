import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { stageFixture } from "./stage-fixture";
import type { AcceptanceDecision } from "../src/components/recommendation-evidence";
import { measuredFixture } from "../src/test/recommendation-fixture";

for (const measured of [false, true]) {
  for (const [outcome, label] of [
    ["confirmed_improvement", "Confirmed improvement"],
    ["promising_candidate", "Promising candidate"],
    ["baseline_retained", "Baseline retained"],
  ] as const) {
    test(`${measured ? "Measured" : "Historical"} ${label} stays consistent through dashboard, download and Benchmark Setup`, async ({
      page,
    }, testInfo) => {
      const baseline = outcome === "baseline_retained";
      const source = baseline ? "baseline" : "candidate";
      const yaml = `experiment:\n  name: recommended-${source}\n`;
      const evidence = measured ? measuredFixture() : undefined;
      if (evidence) {
        Object.assign(
          evidence.metrics[0],
          baseline
            ? { candidate: 0.49, delta: -0.01, ci95: [-0.03, 0.01] }
            : outcome === "confirmed_improvement"
              ? { ci95: [0.02, 0.12] }
              : {},
        );
      }
      const explanation = {
        text: measured
          ? `The ${source} configuration is recommended. Holdout correctness score: 0.5 baseline to ${baseline ? "0.49" : "0.55"} candidate (change ${baseline ? "-0.01" : "+0.05"} score units).`
          : `The ${source} is recommended. These results use synthetic evaluation data.`,
        source: measured ? ("measured" as const) : ("llm" as const),
        prompt_version: measured
          ? "recommendation-measured-v1"
          : "recommendation-explanation-v1",
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
        recommendation_evidence: evidence,
        outcome: outcome.replaceAll("_", " "),
        baseline_retained: baseline,
      });
      const report = {
        acceptance,
        explanation,
        recommendation_evidence: evidence,
        confirmed_improvement: outcome === "confirmed_improvement",
        baseline_retained: baseline,
        limitations: [],
      };
      await page.route("**/api/**", async (route) => {
        const path = new URL(route.request().url()).pathname.replace(
          "/api",
          "",
        );
        let data: unknown = null;
        if (path === "/health") data = { status: "ok" };
        else if (path === "/configs") data = { files: [] };
        else if (path === "/configs/schema")
          data = { stages: [], sections: [] };
        else if (path === "/data/files") data = { files: [] };
        else if (path === "/data/questions") data = { questions: [] };
        else if (path.endsWith("/stages")) data = stages;
        else if (path.endsWith("/recommendation")) data = { ...report, yaml };
        else if (path.endsWith("/export/report")) {
          await route.fulfill({
            body: JSON.stringify(report),
            contentType: "application/json",
            headers: {
              "Content-Disposition": 'attachment; filename="report.json"',
            },
          });
          return;
        } else if (path.endsWith("/configuration"))
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
      await expect(page.getByText(label, { exact: true })).toHaveCount(1);
      await expect(
        page.getByText(explanation.text, { exact: true }),
      ).toHaveCount(1);
      await expect(
        page.getByText(/Paired holdout questions: 6 \/ 6/),
      ).toHaveCount(measured ? 2 : 1);
      if (measured) {
        const recommendation = page.locator("section.card").filter({
          has: page.getByRole("heading", {
            name: "Recommendation",
            exact: true,
          }),
        });
        const table = recommendation.getByRole("table", {
          name: "Recommendation measured comparison",
        });
        await expect(
          table.getByRole("row", { name: /Correctness score/ }),
        ).toContainText(baseline ? "0.49" : "0.55");
        await expect(
          table.getByRole("row", { name: /Estimated cost per query/ }),
        ).toContainText("unknown pricing");
        await expect(
          recommendation.getByText(/Tuning: 24; holdout: 6; human-reviewed: 2/),
        ).toBeVisible();
        await recommendation
          .getByText("Configuration changes (1)", { exact: true })
          .click();
        const changes = recommendation.getByRole("table", {
          name: "Recommendation configuration changes",
        });
        await expect(changes).toContainText("dense");
        await expect(changes).toContainText("bm25");
        const reportLink = page.getByRole("link", { name: "Download report" });
        await expect(reportLink).toHaveAttribute(
          "href",
          "/api/optimizations/session/export/report",
        );
        // Chromium's native <a download> requests bypass route mocks. Verify the
        // linked response through fetch; Python tests cover the real attachment.
        const downloadedReport = await page.evaluate(
          async (href) => (await fetch(href!)).json(),
          await reportLink.getAttribute("href"),
        );
        expect(downloadedReport).toEqual(report);
        if (outcome === "promising_candidate") {
          await recommendation.screenshot({
            path: testInfo.outputPath("measured-recommendation.png"),
          });
        }
      }
      const downloaded = page.waitForEvent("download");
      await page.getByRole("button", { name: "Download YAML" }).click();
      const download = await downloaded;
      expect(download.suggestedFilename()).toBe("recommended.yaml");
      expect(await readFile((await download.path())!, "utf8")).toBe(yaml);
      await page
        .getByRole("button", { name: "Open in Benchmark Setup" })
        .click();
      await expect(page).toHaveURL(/\/benchmark-setup$/);
      await page.getByRole("button", { name: "YAML source" }).click();
      await expect(page.locator(".cm-content")).toContainText(
        `recommended-${source}`,
      );
    });
  }
}
