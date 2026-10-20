// Composite the blue scale onto its white table background before selecting
// black or white text. Comparing the actual luminance avoids unreadable
// intermediate shades caused by a fixed intensity threshold.
export function heatmapCellColors(intensity?: number) {
  if (intensity === undefined)
    return { background: "#f8fafc", color: "#0f172a" };
  const alpha = 0.08 + Math.max(0, Math.min(1, intensity)) * 0.84;
  const rgb = [29, 78, 216].map((channel) =>
    Math.round(channel * alpha + 255 * (1 - alpha)),
  );
  const linear = rgb.map((channel) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  const luminance =
    linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
  const whiteContrast = 1.05 / (luminance + 0.05);
  const blackContrast = (luminance + 0.05) / 0.05;
  return {
    background: `rgb(${rgb.join(", ")})`,
    color: whiteContrast > blackContrast ? "#ffffff" : "#000000",
  };
}
