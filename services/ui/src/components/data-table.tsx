import { useMemo, useState, type ReactNode } from "react";
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type Cell,
  type SortingState,
} from "@tanstack/react-table";
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  Download,
  Search,
} from "lucide-react";
import { Button } from "./ui/button";
import { display, download } from "../lib/utils";
const noWrapColumns: string[] = [];
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
  wrapColumns = noWrapColumns,
  visibleColumns,
  initialPageSize = 10,
}: {
  rows: Record<string, unknown>[];
  name?: string;
  cell?: (key: string, row: Record<string, unknown>) => ReactNode | undefined;
  onRowClick?: (row: Record<string, unknown>) => void;
  wrapColumns?: string[];
  visibleColumns?: string[];
  initialPageSize?: number;
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
      })),
    [keys],
  );
  // Render directly so refreshed row data does not remount expanded cell content.
  const renderCell = (entry: Cell<Record<string, unknown>, unknown>) =>
    cell?.(entry.column.id, entry.row.original) ?? (
      <span
        className={`table-value ${wrapColumns.includes(entry.column.id) ? "table-value-wrap" : ""}`}
        title={display(entry.row.original[entry.column.id])}
      >
        {display(entry.row.original[entry.column.id])}
      </span>
    );
  const table = useReactTable({
    data: rows,
    columns,
    state: {
      sorting,
      globalFilter: search,
      columnVisibility: Object.fromEntries(
        keys.map((key) => [
          key,
          !visibleColumns || visibleColumns.includes(key),
        ]),
      ),
      columnOrder: visibleColumns ?? keys,
    },
    onSortingChange: setSorting,
    onGlobalFilterChange: setSearch,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: initialPageSize } },
  });
  return (
    <div className="data-table">
      <div className="table-toolbar">
        <label className="search-field">
          <Search aria-hidden="true" focusable="false" size={16} />
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
          <Download aria-hidden="true" focusable="false" size={14} />
          Export CSV
        </Button>
      </div>
      <div className="table-scroll">
        <table aria-label={`${name} table`}>
          <thead>
            {table.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((header) => (
                  <th
                    key={header.id}
                    aria-sort={
                      header.column.getIsSorted() === "asc"
                        ? "ascending"
                        : header.column.getIsSorted() === "desc"
                          ? "descending"
                          : "none"
                    }
                  >
                    <button onClick={header.column.getToggleSortingHandler()}>
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext(),
                      )}
                      {header.column.getIsSorted() === "asc" ? (
                        <ArrowUp
                          aria-hidden="true"
                          focusable="false"
                          size={14}
                        />
                      ) : header.column.getIsSorted() === "desc" ? (
                        <ArrowDown
                          aria-hidden="true"
                          focusable="false"
                          size={14}
                        />
                      ) : (
                        <ArrowUpDown
                          aria-hidden="true"
                          focusable="false"
                          size={14}
                        />
                      )}
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
                        {renderCell(entry)}
                      </button>
                    ) : (
                      renderCell(entry)
                    )}
                  </td>
                ))}
              </tr>
            ))}
            {!table.getRowModel().rows.length && (
              <tr>
                <td
                  colSpan={table.getVisibleLeafColumns().length || 1}
                  className="text-center"
                >
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
            {Array.from(new Set([initialPageSize, 10, 25, 50, 100]))
              .sort((a, b) => a - b)
              .map((count) => (
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
            <ChevronLeft aria-hidden="true" focusable="false" size={16} />
          </Button>
          <Button
            variant="outline"
            size="icon"
            aria-label="Next page"
            disabled={!table.getCanNextPage()}
            onClick={() => table.nextPage()}
          >
            <ChevronRight aria-hidden="true" focusable="false" size={16} />
          </Button>
        </div>
      </div>
    </div>
  );
}
