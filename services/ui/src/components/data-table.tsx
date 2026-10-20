import { useMemo, useState, type ReactNode } from "react";
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import {
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  Download,
  Search,
} from "lucide-react";
import { Button } from "./ui/button";
import { display, download } from "../lib/utils";
export function csvText(rows: Record<string, unknown>[], columns: string[]) {
  const quote = (value: unknown) => {
    let text = value == null ? "" : display(value);
    if (/^[=+@\-\t\r]/.test(text)) text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
  };
  return [
    columns.map(quote).join(","),
    ...rows.map((row) => columns.map((key) => quote(row[key])).join(",")),
  ].join("\r\n");
}
export function DataTable({
  rows,
  name = "results",
  cell,
  onRowClick,
}: {
  rows: Record<string, unknown>[];
  name?: string;
  cell?: (key: string, row: Record<string, unknown>) => ReactNode | undefined;
  onRowClick?: (row: Record<string, unknown>) => void;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [search, setSearch] = useState("");
  const keys = useMemo(
    () => Array.from(new Set(rows.flatMap((row) => Object.keys(row)))),
    [rows],
  );
  const columns = useMemo<ColumnDef<Record<string, unknown>>[]>(
    () =>
      keys.map((key) => ({
        id: key,
        accessorFn: (row) =>
          typeof row[key] === "number" ? row[key] : display(row[key]),
        header: key.replaceAll("_", " "),
        cell: (info) =>
          cell?.(key, info.row.original) ?? (
            <span
              className="table-value"
              title={display(info.row.original[key])}
            >
              {display(info.row.original[key])}
            </span>
          ),
      })),
    [keys, cell],
  );
  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting, globalFilter: search },
    onSortingChange: setSorting,
    onGlobalFilterChange: setSearch,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 10 } },
  });
  return (
    <div className="data-table">
      <div className="table-toolbar">
        <label className="search-field">
          <Search size={16} />
          <input
            aria-label={`Search ${name}`}
            placeholder="Search table…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>
        <Button
          variant="outline"
          size="sm"
          onClick={() =>
            download(
              `${name}.csv`,
              csvText(
                table.getFilteredRowModel().rows.map((row) => row.original),
                keys,
              ),
              "text/csv;charset=utf-8",
            )
          }
        >
          <Download size={14} />
          Export CSV
        </Button>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            {table.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((header) => (
                  <th key={header.id}>
                    <button onClick={header.column.getToggleSortingHandler()}>
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext(),
                      )}
                      <ArrowUpDown size={12} />
                      {header.column.getIsSorted() === "asc"
                        ? " ↑"
                        : header.column.getIsSorted() === "desc"
                          ? " ↓"
                          : ""}
                    </button>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((entry, index) => (
                  <td key={entry.id}>
                    {onRowClick && index === 0 ? (
                      <button
                        className="text-link"
                        onClick={() => onRowClick(row.original)}
                      >
                        {flexRender(
                          entry.column.columnDef.cell,
                          entry.getContext(),
                        )}
                      </button>
                    ) : (
                      flexRender(
                        entry.column.columnDef.cell,
                        entry.getContext(),
                      )
                    )}
                  </td>
                ))}
              </tr>
            ))}
            {!table.getRowModel().rows.length && (
              <tr>
                <td colSpan={keys.length || 1} className="text-center">
                  No matching rows.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="table-footer">
        <span>
          {table.getFilteredRowModel().rows.length} rows · Page{" "}
          {table.getState().pagination.pageIndex + 1} of{" "}
          {Math.max(1, table.getPageCount())}
        </span>
        <div className="actions">
          <select
            aria-label="Rows per page"
            value={table.getState().pagination.pageSize}
            onChange={(event) => table.setPageSize(Number(event.target.value))}
          >
            {[10, 25, 50, 100].map((count) => (
              <option key={count} value={count}>
                {count} / page
              </option>
            ))}
          </select>
          <Button
            variant="outline"
            size="icon"
            aria-label="Previous page"
            disabled={!table.getCanPreviousPage()}
            onClick={() => table.previousPage()}
          >
            <ChevronLeft size={16} />
          </Button>
          <Button
            variant="outline"
            size="icon"
            aria-label="Next page"
            disabled={!table.getCanNextPage()}
            onClick={() => table.nextPage()}
          >
            <ChevronRight size={16} />
          </Button>
        </div>
      </div>
    </div>
  );
}
