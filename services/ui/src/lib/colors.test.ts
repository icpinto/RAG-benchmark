import { describe, expect, it } from "vitest";
import { heatmapCellColors } from "./colors";

describe("heatmap readability", () => {
  it("keeps missing values neutral and clamps the scale endpoints", () => {
    expect(heatmapCellColors()).toEqual({
      background: "#f8fafc",
      color: "#0f172a",
    });
    expect(heatmapCellColors(-1)).toEqual(heatmapCellColors(0));
    expect(heatmapCellColors(2)).toEqual(heatmapCellColors(1));
    expect(heatmapCellColors(0).color).toBe("#000000");
    expect(heatmapCellColors(1).color).toBe("#ffffff");
  });

  it("maintains at least 4.5:1 text contrast across the entire blue scale", () => {
    for (let step = 0; step <= 100; step++) {
      const { background, color } = heatmapCellColors(step / 100);
      const rgb = background.match(/\d+/g)!.map(Number);
      const channels = rgb.map((channel) => {
        const value = channel / 255;
        return value <= 0.04045
          ? value / 12.92
          : ((value + 0.055) / 1.055) ** 2.4;
      });
      const light =
        channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
      const contrast =
        color === "#ffffff" ? 1.05 / (light + 0.05) : (light + 0.05) / 0.05;
      expect(contrast, `intensity ${step / 100}`).toBeGreaterThanOrEqual(4.5);
    }
  });
});
