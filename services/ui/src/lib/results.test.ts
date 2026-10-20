import { describe, expect, it } from "vitest";
import { comparisonRows, heatmap, retrievalMetrics } from "./results";
import { csvText } from "../components/data-table";
import { activeRun } from "./api";
import type { Result, Run } from "./types";
const result = (
  case_id: string,
  question: string,
  metrics: Record<string, unknown>,
): Result => ({ case_id, question, metrics, answer: "answer" });
describe("results transformations", () => {
  it("aligns heatmaps by question across cases, averages repeats and preserves missing cells", () => {
    const rows = [
      result("a", "same?", { recall: 0.5 }),
      result("b", "same?", { recall: 1 }),
      result("a", "same?", { recall: 1 }),
      result("b", "other?", {}),
    ];
    const matrix = heatmap(rows, "recall");
    expect(matrix.questions).toEqual(["same?", "other?"]);
    expect(matrix.value("a", "same?")).toBe(0.75);
    expect(matrix.value("a", "other?")).toBeUndefined();
  });
  it("finds arbitrary numeric retrieval metrics without treating booleans as scores", () => {
    expect(
      retrievalMetrics([
        result("a", "q", {
          "recall@5": "0.5",
          retrieval_enabled: true,
          latency: 3,
          custom_ndcg: null,
        }),
      ]),
    ).toEqual(["recall@5"]);
  });
  it("outer joins comparisons and calculates deltas only for shared numeric metrics", () => {
    const runs = [{ id: "one" }, { id: "two" }] as Run[];
    const rows = comparisonRows(runs, {
      one: [result("a", "q", { recall: 0.2, text: "old" })],
      two: [
        result("a", "q", { recall: 0.7, text: "new" }),
        result("b", "q", { recall: 1 }),
      ],
    });
    expect(rows).toHaveLength(2);
    expect(rows[0]["Δ recall (two − one)"]).toBeCloseTo(0.5);
    expect(rows[0]["Δ text (two − one)"]).toBeUndefined();
    expect(rows[1]["one · recall"]).toBeUndefined();
  });
  it("uses full IDs to avoid truncated label collisions", () => {
    const runs = [{ id: "same1234-1" }, { id: "same1234-2" }] as Run[];
    expect(
      Object.keys(
        comparisonRows(runs, {
          "same1234-1": [result("a", "q", { score: 1 })],
          "same1234-2": [result("a", "q", { score: 2 })],
        })[0],
      ),
    ).toContain("same1234-2 · score");
  });
  it("escapes CSV and neutralizes spreadsheet formulas", () => {
    expect(
      csvText(
        [{ answer: "=SUM(1,2)", question: 'a "quote"' }],
        ["answer", "question"],
      ),
    ).toContain('"\'=SUM(1,2)","a ""quote"""');
  });
  it("polls only queued and running statuses", () => {
    expect(activeRun("queued")).toBe(true);
    expect(activeRun("running")).toBe(true);
    expect(activeRun("completed")).toBe(false);
    expect(activeRun("failed")).toBe(false);
  });
});
