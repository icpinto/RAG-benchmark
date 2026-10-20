import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useState } from "react";
import catalogData from "../test/catalog.json";
import type { Catalog } from "../lib/types";
import { configValue, parseConfig, updateConfig } from "../lib/config";
import { Fields, StageEditor } from "./config-fields";
const catalog = catalogData as Catalog;
describe("guided component editor", () => {
  for (const stage of catalog.stages)
    for (const component of stage.components)
      it(`round-trips ${stage.key}/${component.name} without adding defaults`, () => {
        const value =
          stage.mode === "extractors"
            ? [component.name]
            : stage.mode === "single" || stage.mode === "indexer"
              ? { [stage.discriminator]: component.name }
              : [{ [stage.discriminator]: component.name }];
        const text = updateConfig("{}\n", stage.path, value);
        const { doc } = parseConfig(text);
        const change = vi.fn();
        render(<StageEditor stage={stage} doc={doc} onChange={change} />);
        expect(
          screen.getByRole("heading", { name: component.name }),
        ).toBeInTheDocument();
        expect(change).not.toHaveBeenCalled();
        expect(configValue(doc, stage.path)).toEqual(value);
      });
  it("updates an existing alias without inserting a competing key", () => {
    const change = vi.fn();
    render(
      <Fields
        fields={[
          {
            key: "normalize",
            label: "Normalize",
            type: "boolean",
            aliases: ["normalize_embeddings"],
          },
        ]}
        doc={parseConfig("normalize_embeddings: true\n").doc}
        path={[]}
        onChange={change}
      />,
    );
    fireEvent.change(screen.getByLabelText("Normalize"), {
      target: { value: "false" },
    });
    expect(change).toHaveBeenCalledWith(["normalize_embeddings"], false);
  });
  it("retains input focus and all typed characters when an omitted field is first set", () => {
    function Harness() {
      const [text, setText] = useState("experiment: {}\n");
      return (
        <Fields
          fields={[{ key: "name", label: "Experiment name", type: "string" }]}
          doc={parseConfig(text).doc}
          path={["experiment"]}
          onChange={(path, value) => setText(updateConfig(text, path, value))}
        />
      );
    }
    render(<Harness />);
    const input = screen.getByLabelText("Experiment name");
    input.focus();
    fireEvent.change(input, { target: { value: "abc" } });
    expect(screen.getByLabelText("Experiment name")).toBe(input);
    expect(input).toHaveValue("abc");
    expect(input).toHaveFocus();
  });
});
