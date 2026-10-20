export type RecordValue = Record<string, unknown>;
export interface Question {
  question: string;
  expected_answer: string;
  ground_truth_doc_ids: string[];
  ground_truth_chunk_ids: string[];
  expected_citations: string[];
}
export interface DataFile {
  name: string;
  path: string;
  size_bytes: number;
  modified_at: number;
}
export interface Run {
  id: string;
  name: string;
  status: string;
  summary?: RecordValue;
  error?: string;
  config_hash?: string;
  dataset_hash?: string;
  model_identifiers?: RecordValue;
  artifact_dir?: string;
  exports?: RecordValue;
  created_at?: string;
  [key: string]: unknown;
}
export interface Result {
  case_id: string;
  question: string;
  answer: string;
  metrics?: RecordValue;
  case_config?: RecordValue;
  artifact_paths?: RecordValue;
  model_identifiers?: RecordValue;
}
export interface Field {
  key: string;
  label: string;
  type:
    "string" | "number" | "integer" | "boolean" | "object" | "array" | "value";
  default?: unknown;
  required?: boolean;
  help?: string;
  choices?: string[];
  aliases?: string[];
  fields?: Field[];
  items?: Field;
  sweepable?: boolean;
  minimum?: number;
  maximum?: number;
  secret?: boolean;
}
export interface Component {
  name: string;
  fields: Field[];
  help?: string;
}
export interface Stage {
  key: string;
  label: string;
  path: string[];
  mode: "variants" | "extractors" | "indexer" | "single";
  discriminator: string;
  components: Component[];
  help?: string;
}
export interface Catalog {
  version: number;
  metric_components?: Component[];
  sections: { key: string; label: string; fields: Field[] }[];
  stages: Stage[];
}
