export interface AcceptanceDecision {
  outcome:
    "confirmed_improvement" | "promising_candidate" | "baseline_retained";
  recommended_source: "baseline" | "candidate";
  reason_codes: string[];
  checks: {
    code: string;
    label: string;
    status: "pass" | "fail" | "unavailable";
    measured: unknown;
    threshold: unknown;
    required: boolean;
  }[];
}

export interface RecommendationExplanation {
  text: string;
  source: "llm" | "fallback";
  prompt_version: string;
  fallback_reason?: string;
}

export const recommendationLabel = (outcome: AcceptanceDecision["outcome"]) =>
  ({
    confirmed_improvement: "Confirmed improvement",
    promising_candidate: "Promising candidate",
    baseline_retained: "Baseline retained",
  })[outcome];

export function RecommendationEvidence({
  acceptance,
}: {
  acceptance?: AcceptanceDecision;
}) {
  const coverage = acceptance?.checks.find((c) => c.code === "paired_coverage")
    ?.measured as
    | { paired: number | null; expected: number; fraction: number | null }
    | undefined;
  return (
    <>
      {coverage && (
        <p>
          Paired holdout questions: {coverage.paired ?? "unavailable"} /{" "}
          {coverage.expected}
          {coverage.fraction != null &&
            ` (${(coverage.fraction * 100).toFixed(1)}%)`}
          ; required coverage: 95%.
        </p>
      )}
      {acceptance?.checks.some((c) => c.status !== "pass") && (
        <ul aria-label="Acceptance checks not passed">
          {acceptance.checks
            .filter((c) => c.status !== "pass")
            .map((c) => (
              <li key={c.code}>
                {c.status === "unavailable" ? "Unavailable" : "Not met"}:{" "}
                {c.label}
              </li>
            ))}
        </ul>
      )}
    </>
  );
}
