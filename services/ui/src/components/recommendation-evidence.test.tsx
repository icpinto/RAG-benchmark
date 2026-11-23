import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  RecommendationEvidence,
  RecommendationResults,
} from "./recommendation-evidence";
import {
  measuredAcceptance,
  measuredFixture,
} from "../test/recommendation-fixture";

describe("measured recommendation results", () => {
  it("shows scores, signed changes, confidence bounds, real zeros and unknown pricing", () => {
    render(
      <RecommendationResults
        evidence={measuredFixture()}
        acceptance={measuredAcceptance()}
      />,
    );
    const table = within(
      screen.getByRole("table", { name: "Recommendation measured comparison" }),
    );
    const correctness = within(
      table.getByRole("row", { name: /Correctness score/ }),
    );
    for (const value of ["0.5", "0.55", "+0.05", "-0.02 to +0.12", "6"]) {
      expect(correctness.getByText(value, { exact: true })).toBeInTheDocument();
    }
    expect(
      table.getByRole("row", { name: /p95 query latency \(seconds\)/ }),
    ).toHaveTextContent("-0.4");
    const cost = table.getByRole("row", {
      name: /Estimated cost per query \(USD\/query\)/,
    });
    expect(within(cost).getByText("0", { exact: true })).toBeInTheDocument();
    expect(cost).toHaveTextContent("unknown pricing");
    expect(
      table.getByRole("row", { name: /Citation support score/ }),
    ).toHaveTextContent("paired scores unavailable");
    expect(
      screen.getByText(/Paired holdout questions: 6 \/ 6 \(100.0%\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Lower bound of the 95% correctness interval: -0.02; required: > 0/,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Tuning: 24; holdout: 6; human-reviewed: 2/),
    ).toBeInTheDocument();
    expect(screen.getByText("Configuration changes (1)")).toBeInTheDocument();
    const changes = within(
      screen.getByRole("table", {
        name: "Recommendation configuration changes",
        hidden: true,
      }),
    );
    expect(changes.getByText("dense")).toBeInTheDocument();
    expect(changes.getByText("bm25")).toBeInTheDocument();
  });

  it("keeps tiny changes visible and renders configuration strings as text", () => {
    const evidence = measuredFixture();
    evidence.metrics[0].delta = 1e-10;
    evidence.metrics[0].ci95 = [1e-12, 2e-10];
    evidence.configuration_differences[0].candidate =
      "<script>untrusted</script>";
    evidence.evaluation.confirmation_stop_reason = "Runtime budget exhausted";
    render(<RecommendationResults evidence={evidence} />);
    expect(screen.getByText("+1e-10", { exact: true })).toBeInTheDocument();
    expect(screen.getByText("+1e-12 to +2e-10")).toBeInTheDocument();
    const text = screen.getByText("<script>untrusted</script>");
    expect(text.querySelector("script")).toBeNull();
    expect(
      screen.getByText(/Confirmation stop reason: Runtime budget exhausted/),
    ).toBeInTheDocument();
  });

  it("shares measured failure details with Confirmation while retaining legacy labels", () => {
    const acceptance = measuredAcceptance();
    const evidence = measuredFixture();
    acceptance.checks.push({
      code: "query_latency",
      label: "Candidate meets latency limit",
      status: "fail",
      measured: 1.25,
      threshold: 0.5,
      required: true,
    });
    evidence.check_details.query_latency =
      "Candidate p95 query latency: 1.25 s; limit: 0.5 s.";
    const { rerender } = render(
      <RecommendationEvidence acceptance={acceptance} evidence={evidence} />,
    );
    expect(
      screen.getByText(/Candidate p95 query latency: 1.25 s; limit: 0.5 s/),
    ).toBeInTheDocument();
    rerender(<RecommendationEvidence acceptance={acceptance} />);
    expect(
      screen.getByText(/Not met: Candidate meets latency limit/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/limit: 0.5 s/)).not.toBeInTheDocument();
  });
});
