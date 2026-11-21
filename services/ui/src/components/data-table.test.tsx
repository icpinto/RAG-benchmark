import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { DataTable } from "./data-table";
import * as utils from "../lib/utils";

it("sorts rows in both directions and exposes the sort state without changing the control name", async () => {
  const user = userEvent.setup();
  render(<DataTable rows={[{ name: "Bravo" }, { name: "Alpha" }]} />);
  const table = screen.getByRole("table");
  const header = within(table).getByRole("columnheader", { name: "name" });
  const sort = within(header).getByRole("button", { name: "name" });
  const values = () =>
    within(table)
      .getAllByRole("cell")
      .map((cell) => cell.textContent);
  const expectIcon = () => {
    const icons = sort.querySelectorAll("svg");
    expect(icons).toHaveLength(1);
    expect(icons[0]).toHaveAttribute("aria-hidden", "true");
    expect(icons[0]).toHaveAttribute("focusable", "false");
    expect(sort).toHaveAccessibleName("name");
  };

  expect(header).toHaveAttribute("aria-sort", "none");
  expectIcon();
  await user.click(sort);
  expect(header).toHaveAttribute("aria-sort", "ascending");
  expect(values()).toEqual(["Alpha", "Bravo"]);
  expectIcon();
  await user.click(sort);
  expect(header).toHaveAttribute("aria-sort", "descending");
  expect(values()).toEqual(["Bravo", "Alpha"]);
  expectIcon();
  await user.click(sort);
  expect(header).toHaveAttribute("aria-sort", "none");
  expect(values()).toEqual(["Bravo", "Alpha"]);
  expectIcon();
});

it("pages compact columns while searching and exporting complete rows", async () => {
  const user = userEvent.setup();
  const download = vi.spyOn(utils, "download").mockImplementation(() => {});
  render(
    <DataTable
      name="compact"
      rows={Array.from({ length: 7 }, (_, index) => ({
        name: `Trial ${index + 1}`,
        hidden_setting: index === 6 ? "unique embedding" : "dense",
        score: index,
      }))}
      visibleColumns={["name", "score"]}
      initialPageSize={5}
    />,
  );
  const table = screen.getByRole("table");
  expect(within(table).getAllByRole("columnheader")).toHaveLength(2);
  expect(within(table).getAllByRole("row")).toHaveLength(6);
  expect(screen.getByRole("combobox", { name: "Rows per page" })).toHaveValue(
    "5",
  );
  await user.click(screen.getByRole("button", { name: "Export CSV" }));
  expect(download).toHaveBeenLastCalledWith(
    "compact.csv",
    expect.stringContaining('"Trial 7","unique embedding","6"'),
    "text/csv;charset=utf-8",
  );
  await user.click(screen.getByRole("button", { name: "Next page" }));
  expect(within(table).getAllByRole("row")).toHaveLength(3);
  await user.type(
    screen.getByRole("textbox", { name: "Search compact" }),
    "unique embedding",
  );
  expect(await within(table).findByText("Trial 7")).toBeInTheDocument();
  expect(within(table).getAllByRole("row")).toHaveLength(2);
  await user.click(screen.getByRole("button", { name: "Export CSV" }));
  expect(download).toHaveBeenLastCalledWith(
    "compact.csv",
    '"name","hidden_setting","score"\r\n"Trial 7","unique embedding","6"',
    "text/csv;charset=utf-8",
  );
});
