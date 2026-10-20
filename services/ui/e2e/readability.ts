import { expect, type Page } from "@playwright/test";

export async function expectReadableLayout(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
    "The page should fit the viewport; tables scroll inside their panels",
  ).toBe(true);
  const failures = await page.evaluate(() => {
    const rgba = (value: string) => value.match(/[\d.]+/g)!.map(Number);
    const luminance = (rgb: number[]) => {
      const linear = rgb.slice(0, 3).map((value) => {
        const channel = value / 255;
        return channel <= 0.04045
          ? channel / 12.92
          : ((channel + 0.055) / 1.055) ** 2.4;
      });
      return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
    };
    const background = (element: HTMLElement) => {
      const ancestors: HTMLElement[] = [];
      for (
        let item: HTMLElement | null = element;
        item;
        item = item.parentElement
      )
        ancestors.unshift(item);
      return ancestors.reduce(
        (under, item) => {
          const color = rgba(getComputedStyle(item).backgroundColor);
          const alpha = color[3] ?? 1;
          return under.map(
            (channel, index) => color[index] * alpha + channel * (1 - alpha),
          );
        },
        [255, 255, 255],
      );
    };
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
    );
    const errors: string[] = [];
    while (walker.nextNode()) {
      const text = walker.currentNode.textContent?.trim();
      const element = walker.currentNode.parentElement;
      if (
        !text ||
        !element ||
        !element.getClientRects().length ||
        element.closest("[inert], script, style")
      )
        continue;
      const style = getComputedStyle(element);
      if (style.visibility !== "visible") continue;
      const foreground = luminance(rgba(style.color));
      const behind = luminance(background(element));
      const contrast =
        (Math.max(foreground, behind) + 0.05) /
        (Math.min(foreground, behind) + 0.05);
      if (contrast < 4.5 || parseFloat(style.fontSize) < 12)
        errors.push(
          `${text.slice(0, 60)}: contrast ${contrast.toFixed(2)}, ${style.fontSize}`,
        );
    }
    return errors;
  });
  expect(
    failures,
    "Visible text should meet the contrast and minimum font size targets",
  ).toEqual([]);
}
