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
  source: "llm" | "fallback" | "measured";
  prompt_version: string;
  fallback_reason?: string;
}

export interface RecommendationMeasurements {
  version: number;
  baseline_trial_id: string | null;
  candidate_trial_id: string | null;
  metrics: {
    metric: string;
    label: string;
    unit: "score" | "seconds" | "USD/query";
    aggregation: "paired_mean" | "p95" | "mean";
    baseline: number | null;
    candidate: number | null;
    delta: number | null;
    ci95: number[] | null;
    paired_count: number | null;
    baseline_reason: string | null;
    candidate_reason: string | null;
  }[];
  coverage: {
    paired: number | null;
    expected: number;
    fraction: number | null;
    required_fraction: number;
  };
  check_details: Record<string, string>;
  configuration_differences: {
    setting: string;
    baseline: unknown;
    candidate: unknown;
  }[];
  evaluation: {
    synthetic_data: boolean;
    dataset_provenance: {
      status: "generated" | "reused";
      dataset_id: string;
      source_dataset_id: string;
      source_session_id: string | null;
      question_count: number;
    } | null;
    split_counts: { tuning: number; holdout: number };
    human_reviewed_questions: number;
    confirmation_stop_reason: string | null;
  };
}

const valueText = (value: number | null | undefined, signed = false) => {
  if (value == null || !Number.isFinite(value)) return "Unavailable";
  const text = Number(value.toPrecision(6)).toString();
  return signed && value > 0 ? `+${text}` : text;
};

export const recommendationLabel = (outcome: AcceptanceDecision["outcome"]) =>
  ({
    confirmed_improvement: "Confirmed improvement",
    promising_candidate: "Promising candidate",
    baseline_retained: "Baseline retained",
  })[outcome];

export function RecommendationEvidence({
  acceptance,
  evidence,
}: {
  acceptance?: AcceptanceDecision;
  evidence?: RecommendationMeasurements;
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
          ; required coverage:{" "}
          {evidence
            ? valueText(evidence.coverage.required_fraction * 100)
            : "95"}
          %.
        </p>
      )}
      {acceptance?.checks.some((c) => c.status !== "pass") && (
        <ul aria-label="Acceptance checks not passed">
          {acceptance.checks
            .filter((c) => c.status !== "pass")
            .map((c) => (
              <li key={c.code}>
                {c.status === "unavailable" ? "Unavailable" : "Not met"}:{" "}
                {evidence?.check_details[c.code] ?? c.label}
              </li>
            ))}
        </ul>
      )}
    </>
  );
}

export function RecommendationResults({
  evidence,
  acceptance,
}: {
  evidence: RecommendationMeasurements;
  acceptance?: AcceptanceDecision;
}) {
  const evaluation = evidence.evaluation;
  const provenance = evaluation.dataset_provenance;
  const settingText = (value: unknown) =>
    value == null
      ? "Not set"
      : typeof value === "string"
        ? value
        : JSON.stringify(value);
  return (
    <>
      <div className="table-scroll">
        <table aria-label="Recommendation measured comparison">
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col">Baseline</th>
              <th scope="col">Candidate</th>
              <th scope="col" title="Candidate minus baseline">
                Change
              </th>
              <th scope="col">95% interval for change</th>
              <th scope="col">Paired questions</th>
            </tr>
          </thead>
          <tbody>
            {evidence.metrics.map((metric) => (
              <tr key={metric.metric}>
                <th scope="row">
                  {metric.label}
                  {metric.unit !== "score" && ` (${metric.unit})`}
                </th>
                <td
                  title={
                    metric.baseline == null
                      ? undefined
                      : String(metric.baseline)
                  }
                >
                  {metric.baseline == null
                    ? (metric.baseline_reason ?? "Unavailable")
                    : valueText(metric.baseline)}
                </td>
                <td
                  title={
                    metric.candidate == null
                      ? undefined
                      : String(metric.candidate)
                  }
                >
                  {metric.candidate == null
                    ? (metric.candidate_reason ?? "Unavailable")
                    : valueText(metric.candidate)}
                </td>
                <td
                  title={
                    metric.delta == null ? undefined : String(metric.delta)
                  }
                >
                  {valueText(metric.delta, true)}
                </td>
                <td>
                  {metric.aggregation !== "paired_mean"
                    ? "—"
                    : metric.ci95?.length === 2
                      ? metric.ci95
                          .map((bound) => valueText(bound, true))
                          .join(" to ")
                      : "Unavailable"}
                </td>
                <td>
                  {metric.aggregation === "paired_mean"
                    ? valueText(metric.paired_count)
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted">
        Changes are candidate minus baseline. Quality scores use paired holdout
        questions. Latency is p95 across observed confirmation queries;
        estimated cost is the mean per query in USD.
      </p>
      <RecommendationEvidence acceptance={acceptance} evidence={evidence} />
      <details>
        <summary>
          Configuration changes ({evidence.configuration_differences.length})
        </summary>
        {evidence.configuration_differences.length ? (
          <div className="table-scroll">
            <table aria-label="Recommendation configuration changes">
              <thead>
                <tr>
                  <th scope="col">Setting</th>
                  <th scope="col">Baseline</th>
                  <th scope="col">Candidate</th>
                </tr>
              </thead>
              <tbody>
                {evidence.configuration_differences.map((change) => (
                  <tr key={change.setting}>
                    <th scope="row">{change.setting}</th>
                    <td className="table-value-wrap">
                      {settingText(change.baseline)}
                    </td>
                    <td className="table-value-wrap">
                      {settingText(change.candidate)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No configuration changes recorded.</p>
        )}
      </details>
      <p>
        {evaluation.synthetic_data
          ? "Synthetic evaluation dataset"
          : "Evaluation dataset"}
        {provenance
          ? ` (${provenance.status}, ${provenance.question_count} questions)`
          : ""}
        . Tuning: {evaluation.split_counts.tuning}; holdout:{" "}
        {evaluation.split_counts.holdout}; human-reviewed:{" "}
        {evaluation.human_reviewed_questions}.
      </p>
      {provenance && (
        <details>
          <summary>Dataset provenance</summary>
          <p>
            Dataset: {provenance.dataset_id}. Source dataset:{" "}
            {provenance.source_dataset_id}.
            {provenance.source_session_id &&
              ` Source session: ${provenance.source_session_id}.`}
          </p>
        </details>
      )}
      {evaluation.confirmation_stop_reason && (
        <p>Confirmation stop reason: {evaluation.confirmation_stop_reason}</p>
      )}
    </>
  );
}
