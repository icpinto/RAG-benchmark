import { test, expect } from "@playwright/test";
import { stageFixture } from "./stage-fixture";
import { expectReadableLayout } from "./readability";

for (const width of [1440, 375]) {
  for (const version of [6, 7]) {
    test(`inspect DSPy refinement and confirmation v${version} at ${width}px`, async ({
      page,
    }) => {
      const stages = stageFixture([
        { id: "answer-winner", phase: "answer", status: "completed" },
      ]);
      stages.workflow_version = 6;
      stages.stages.prompt = {
        status: "completed",
        split: "prompt validation",
        question_count: 6,
        primary_metric: "answer_correctness",
        trials: [],
        parent_answer_trial_id: "answer-winner",
        budget_required: { tokens: 4000000, seconds: 1800, cost: null },
        plan: [
          {
            candidate_hash: "first",
            source_trial_id: null,
            variant: "dspy:mipro_zero_shot_v1",
            scheduled: true,
            status: "failed",
            trial_id: null,
            error: "Proposal model timed out",
          },
          {
            candidate_hash: "second",
            source_trial_id: null,
            variant: "dspy:bootstrap_small_v1",
            scheduled: true,
            status: "completed",
            trial_id: "compiled-trial",
            artifact_id: "saved-artifact",
          },
        ],
      };
      stages.stages.confirm.selected_trial_id = "compiled-trial";
      stages.stages.confirm.selected_phase = "prompt";
      const control = stages.stages.answer.trials[0];
      stages.stages.prompt.trials = [
        {
          ...control,
          roles: ["unchanged answer winner"],
          split: "prompt validation",
        },
        {
          ...control,
          id: "compiled-trial",
          phase: "prompt",
          roles: ["selected for holdout"],
          split: "prompt validation",
          source_answer_trial_id: control.id,
          metrics: {
            ...control.metrics,
            answer_correctness: {
              ...control.metrics.answer_correctness,
              mean: 0.95,
            },
          },
          ranking: { ...control.ranking!, score: 0.95 },
          configuration: {
            ...control.configuration,
            prompt: { type: "compiled" },
          },
        },
      ];
      stages.stages.prompt.compilations = [
        {
          id: "failed-job",
          status: "failed",
          progress: { current_config: 0 },
          usage: { tokens: 100, cost: null },
        },
        {
          id: "compiled-job",
          status: "completed",
          progress: { current_config: 0 },
          usage: { tokens: 200, cost: null },
        },
      ];
      if (version === 7) {
        const prompt = stages.stages.prompt;
        stages.workflow_version = 7;
        stages.stages.answer.trials = prompt.trials.map((trial) => ({
          ...trial,
          phase: "answer",
        }));
        stages.stages.answer.compilations = prompt.compilations;
        stages.stages.answer.refinement = {
          status: "completed",
          shortlist: [{ trial_id: "answer-winner", parent_hash: "parent" }],
          budget_required: prompt.budget_required,
          plan: prompt.plan!.map((entry) => ({
            ...entry,
            parent_answer_trial_id: "answer-winner",
          })),
        };
        stages.stages.confirm.selected_phase = "answer";
        delete stages.stages.prompt;
      }
      await page.route("**/api/**", async (route) => {
        const path = new URL(route.request().url()).pathname.replace(
          "/api",
          "",
        );
        let data: unknown = [];
        if (path === "/health") data = { status: "ok" };
        else if (path === "/configs" || path === "/data/files")
          data = { files: [] };
        else if (path === "/optimizations/session")
          data = {
            id: "session",
            status: "completed",
            stage: "done",
            progress: {},
            budget: { tokens: 100, active_seconds: 10, cost: null },
          };
        else if (path.endsWith("/stages")) data = stages;
        else if (path.endsWith("/configuration"))
          data = { request: { limits: {} }, template: {} };
        else if (path.endsWith("/recommendation") || path.endsWith("/dataset"))
          data = null;
        await route.fulfill({ json: data });
      });
      await page.setViewportSize({ width, height: 1000 });
      await page.goto("/auto-optimize/session?stage=answer");
      if (version === 6) {
        await expect(page.getByRole("tab", { name: /DSPy/ })).toBeVisible();
        await page.getByRole("tab", { name: /DSPy/ }).click();
        await expect(page).toHaveURL(/stage=prompt/);
      } else {
        await expect(page.getByRole("tab")).toHaveCount(3);
        await expect(
          page.getByRole("heading", { name: "DSPy prompt refinement" }),
        ).toBeVisible();
      }
      await expect(page.getByText("Proposal model timed out")).toBeVisible();
      if (version === 6)
        await expect(page.getByText("saved-artifact")).toBeVisible();
      await expectReadableLayout(page);
      await page.screenshot({
        path: `test-results/dspy-refinement-v${version}-${width}.png`,
        fullPage: true,
      });
      await page.getByRole("tab", { name: /Confirmation/ }).click();
      await expect(
        page.getByText(
          `Candidate selected from ${version === 6 ? "DSPy" : "answer"} trial`,
        ),
      ).toBeVisible();
      await page.getByRole("link", { name: "compiled" }).click();
      await expect(page).toHaveURL(
        version === 6
          ? /stage=prompt#trial-compiled-trial/
          : /stage=answer#trial-compiled-trial/,
      );
    });
  }
}
