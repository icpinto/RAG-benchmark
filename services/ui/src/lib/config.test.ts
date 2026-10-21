import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseConfig, configValue, updateConfig } from "./config";

describe("configuration round trips", () => {
  const sample = readFileSync(
    resolve(process.cwd(), "../../configs/default.yaml"),
    "utf8",
  );
  it("edits the effective final runtime without merging duplicate mappings", () => {
    const duplicated = sample + "\nruntime:\n  max_combinations: 32\n";
    const original = parseConfig(duplicated);
    expect(original.errors).toEqual([]);
    expect(original.duplicates).toContain("runtime");
    expect(configValue(original.doc, ["runtime", "pricing"])).toBeUndefined();
    const edited = updateConfig(duplicated, ["runtime", "max_combinations"], 2);
    expect(
      configValue(parseConfig(edited).doc, ["runtime", "max_combinations"]),
    ).toBe(2);
    expect(
      configValue(parseConfig(edited).doc, ["runtime", "pricing"]),
    ).toBeUndefined();
    expect(edited.match(/^runtime:/gm)).toHaveLength(2);
    expect(edited).toContain("# Prefer docling");
    expect(edited).toContain("${OLLAMA_MODEL:gpt-oss:20b}");
  });
  it("preserves plugins, aliases, unknown keys and untouched scalar/list forms", () => {
    const source =
      'custom: &custom {x: 1}\nplugin: *custom\nllm: {model: "${MODEL:local}"}\npipeline:\n  ingestion:\n    chunkers: [{type: fixed, chunk_size: [128, 256], overlap: 0.2, custom: yes}]\n';
    const edited = updateConfig(
      source,
      ["pipeline", "ingestion", "chunkers", 0, "overlap"],
      [0, 0.1],
    );
    const doc = parseConfig(edited).doc;
    expect(configValue(doc, ["plugin"])).toEqual({ x: 1 });
    expect(
      configValue(doc, ["pipeline", "ingestion", "chunkers", 0, "chunk_size"]),
    ).toEqual([128, 256]);
    expect(
      configValue(doc, ["pipeline", "ingestion", "chunkers", 0, "custom"]),
    ).toBe(true);
    expect(configValue(doc, ["llm", "model"])).toBe("${MODEL:local}");
  });
  it("editing an aliased component does not change the anchor source", () => {
    const edited = updateConfig(
      "base: &base {type: fixed, chunk_size: 128}\nvariants: [*base]\n",
      ["variants", 0, "chunk_size"],
      256,
    );
    const doc = parseConfig(edited).doc;
    expect(configValue(doc, ["base", "chunk_size"])).toBe(128);
    expect(configValue(doc, ["variants", 0, "chunk_size"])).toBe(256);
  });
  it("reports invalid YAML and never overwrites the draft", () => {
    const source = "pipeline: [broken";
    expect(parseConfig(source).errors.length).toBeGreaterThan(0);
    expect(() =>
      updateConfig(source, ["runtime", "max_combinations"], 4),
    ).toThrow();
    expect(parseConfig("[1, 2]").errors[0]).toContain("mapping");
  });
  it("adds and removes only requested parameters without injecting defaults", () => {
    const edited = updateConfig(
      "experiment: {name: test}\n",
      ["runtime", "embedding_cache", "enabled"],
      true,
    );
    const reset = updateConfig(
      edited,
      ["runtime", "embedding_cache", "enabled"],
      undefined,
    );
    expect(
      configValue(parseConfig(reset).doc, ["runtime", "embedding_cache"]),
    ).toEqual({});
    expect(
      configValue(parseConfig(reset).doc, ["runtime", "max_combinations"]),
    ).toBeUndefined();
  });
  it("retains comments when adding component variants and leaves later aliases intact", () => {
    const source =
      "base: &base {chunk_size: 128}\nvariants:\n  # Keep this explanation\n  - type: fixed # Important variant\n    chunk_size: 128\nreferences: [*base]\n";
    const edited = updateConfig(
      source,
      ["variants"],
      [{ type: "fixed", chunk_size: 128 }, { type: "recursive" }],
    );
    expect(edited).toContain("# Important variant");
    expect(edited).toContain("# Keep this explanation");
    expect(
      configValue(parseConfig(edited).doc, ["references", 0, "chunk_size"]),
    ).toBe(128);
  });
  it("keeps downstream references to the original anchor after editing an alias", () => {
    const edited = updateConfig(
      "base: &base {type: fixed, chunk_size: 128}\nvariants: [*base]\nlater: *base\n",
      ["variants", 0, "chunk_size"],
      256,
    );
    expect(configValue(parseConfig(edited).doc, ["later", "chunk_size"])).toBe(
      128,
    );
  });
});
