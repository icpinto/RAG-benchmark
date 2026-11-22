import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it } from "vitest";
import { DataTable } from "./data-table";
import { useState } from "react";

it("preserves an open custom inspector when polled data and cell callbacks update", async () => {
  const user = userEvent.setup();
  function Inspector({ value }: { value: unknown }) {
    const [open, setOpen] = useState(false);
    return (
      <>
        <button onClick={() => setOpen(!open)}>Inspect</button>
        {open && <p>Measured score: {String(value)}</p>}
      </>
    );
  }
  const { rerender } = render(
    <DataTable
      rows={[{ score: 0.5 }]}
      cell={(_, row) => <Inspector value={row.score} />}
    />,
  );
  await user.click(screen.getByRole("button", { name: "Inspect" }));
  expect(screen.getByText("Measured score: 0.5")).toBeVisible();
  rerender(
    <DataTable
      rows={[{ score: 0.8 }]}
      cell={(_, row) => <Inspector value={row.score} />}
    />,
  );
  expect(screen.getByText("Measured score: 0.8")).toBeVisible();
});

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
