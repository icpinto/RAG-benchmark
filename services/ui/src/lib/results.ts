import type { RecordValue, Result, Run } from "./types";
export function metricRows(results: Result[]): RecordValue[] {
  return results.map((item) => ({
    case_id: item.case_id,
    question: item.question,
    answer: item.answer,
    ...(item.metrics || {}),
  }));
}
export function numeric(value: unknown): number | undefined {
  if (typeof value === "number")
    return Number.isFinite(value) ? value : undefined;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}
export function retrievalMetrics(results: Result[]) {
  return Array.from(
    new Set(
      results.flatMap((row) =>
        Object.entries(row.metrics || {})
          .filter(
            ([key, value]) =>
              /recall|precision|mrr|ndcg|hit|retrieval|citation/i.test(key) &&
              numeric(value) !== undefined,
          )
          .map(([key]) => key),
      ),
    ),
  );
}
export function heatmap(results: Result[], metric: string) {
  const cases = Array.from(new Set(results.map((row) => row.case_id)));
  const questions = Array.from(new Set(results.map((row) => row.question)));
  const cells = new Map<string, number[]>();
  for (const row of results) {
    const value = numeric(row.metrics?.[metric]);
    if (value !== undefined) {
      const key = JSON.stringify([row.case_id, row.question]);
      cells.set(key, [...(cells.get(key) || []), value]);
    }
  }
  return {
    cases,
    questions,
    value: (caseId: string, question: string) => {
      const values = cells.get(JSON.stringify([caseId, question]));
      return values?.length
        ? values.reduce((a, b) => a + b, 0) / values.length
        : undefined;
    },
  };
}
export function comparisonRows(
  runs: Run[],
  results: Record<string, Result[]>,
): RecordValue[] {
  const joined = new Map<string, RecordValue>();
  for (const run of runs)
    for (const row of results[run.id] || []) {
      const key = JSON.stringify([row.case_id, row.question]);
      const output = joined.get(key) || {
        case_id: row.case_id,
        question: row.question,
      };
      for (const [metric, value] of Object.entries(row.metrics || {}))
        output[`${run.id} · ${metric}`] = value;
      joined.set(key, output);
    }
  const baseline = runs[0];
  if (baseline)
    for (const row of joined.values())
      for (const run of runs.slice(1))
        for (const [key, value] of Object.entries(row)) {
          const prefix = `${run.id} · `;
          if (key.startsWith(prefix)) {
            const metric = key.slice(prefix.length);
            const base = row[`${baseline.id} · ${metric}`];
            if (
              typeof value === "number" &&
              typeof base === "number" &&
              Number.isFinite(value) &&
              Number.isFinite(base)
            )
              row[`Δ ${metric} (${run.id} − ${baseline.id})`] = value - base;
          }
        }
  return Array.from(joined.values());
}
export function aggregateRows(run: Run): { metric: string; value: unknown }[] {
  const rows: { metric: string; value: unknown }[] = [];
  function flatten(value: unknown, path: string) {
    if (value && typeof value === "object" && !Array.isArray(value))
      Object.entries(value).forEach(([key, child]) =>
        flatten(child, path ? `${path}.${key}` : key),
      );
    else rows.push({ metric: path, value });
  }
  flatten(run.summary?.aggregate || {}, "");
  return rows;
}
