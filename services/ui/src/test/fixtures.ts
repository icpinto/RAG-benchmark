import type { Result, Run } from "../lib/types";
export const sampleConfig = `experiment:\n  name: research_run\ndata:\n  input_dir: /app/data/raw\n  ground_truth_file: /app/data/eval/qa.json\npipeline:\n  ingestion:\n    extractors: [txt]\n    chunkers: [{type: fixed, chunk_size: 256}]\n    embeddings: [{provider: sentence_transformers, model: local}]\n  query:\n    retrievers: [{type: dense}]\nllm:\n  provider: ollama\n  model: local\nmetrics:\n  retrieval_k: [3, 5]\nruntime:\n  max_combinations: 1\n`;
export const sampleRuns: Run[] = [
  {
    id: "run-one",
    name: "Research baseline",
    status: "completed",
    created_at: "2026-10-20T10:00:00Z",
    config_hash: "config-one",
    dataset_hash: "dataset-one",
    model_identifiers: { llms: ["ollama:local"] },
    summary: {
      cases: 1,
      questions: 1,
      errors: 0,
      aggregate: { "recall@5": { mean: 0.5, min: 0.5, max: 0.5, sum: 0.5 } },
    },
  },
  {
    id: "run-two",
    name: "Research candidate",
    status: "completed",
    config_hash: "config-two",
    dataset_hash: "dataset-one",
    summary: {
      cases: 1,
      questions: 1,
      errors: 0,
      aggregate: { "recall@5": { mean: 1 } },
    },
  },
];
export const sampleResults: Result[] = [
  {
    case_id: "case-one",
    question: "What is the refund window?",
    answer: "The refund window is 14 days.",
    metrics: { "recall@5": 0.5, latency_ms: 123 },
    case_config: {
      evaluation_step: {
        context: "Refunds are accepted within 14 days.",
        rewritten_queries: ["refund window"],
        query_build: "Answer using the refund policy.",
        retrieved_context: [
          {
            rank: 1,
            score: 0.92,
            retriever: "dense",
            doc_id: "policy.md",
            chunk_id: "chunk-1",
            text: "Refunds are accepted within 14 days.",
            metadata: { source: "policy.md" },
          },
        ],
      },
    },
  },
];
