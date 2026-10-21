# React workspace

The UI uses React/TypeScript, Vite, Tailwind, shadcn-style Radix components,
TanStack Query/Table, CodeMirror, and Recharts. The three workspaces call the
existing FastAPI API through the same-origin `/api` proxy.

## Development

Use Node 22.12 or later. Start the backend at port 7800, then:

```bash
cd services/ui
npm ci
npm run dev
```

Open http://localhost:5173. `UI_API_PROXY_TARGET` overrides the development
proxy target (default `http://localhost:7800`); it is a server environment
variable, never included in browser code. `VITE_API_DOCS_URL` optionally
overrides the documentation link; its default uses the current hostname on
port 7800. Never put provider keys in `VITE_` variables.

## Production

From the repository root, run `docker compose up --build`. The UI is served at
http://localhost:8499 by Nginx. The runtime `API_BASE_URL` points to the API
service and must omit a trailing slash (default `http://api:8000`). Nginx strips
`/api` before forwarding, supports multipart uploads and direct SPA links, and
does not mount configurations, data, or artifacts into the UI container.

## Configuration forms

`GET /configs/schema` is a read-only, provider-independent catalog maintained
alongside the API. It describes built-in fields, aliases, nested settings,
defaults, choices, and the paths expanded as parameter sweeps. Defaults shown
in forms are hints and are only written when explicitly chosen.

Forms and YAML share one in-memory draft. The YAML document AST retains
comments, unknown components/fields, anchors, and environment expressions.
Duplicate mapping keys use the backend's last-occurrence semantics. Invalid
YAML pauses forms until corrected. Drafts and credentials are not stored in
browser storage. Loading a different file or leaving setup prompts before
discarding edits.

Extractor-specific settings can be saved in `pipeline.ingestion.extractor_configs`;
the case expander forwards them to execution.
Metrics `enabled` selects the metrics computed, and the engine uses the first context
assembler, prompt builder, and citation validator. These engine behaviors
are retained and explained in the relevant forms.

## Checks

```bash
npm run typecheck
npm run lint
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

Browser tests run against a local Vite server with mocked API responses;
they do not require provider keys or run model inference. They cover uploads,
question editing, save/launch order, status polling, traces, comparisons,
mobile navigation, error recovery, and direct-link refresh. Unit tests exercise
every built-in form family and YAML compatibility. Run the Python suite from
the repository root using `pytest`.

After changing the catalog, run `python scripts/export_ui_catalog.py` from the
repository root and rerun both suites. The Python contract test detects stale
frontend fixtures. `npm run format` formats
the frontend source.

Runs ending with `completed_with_errors` are terminal and display a warning badge.
Compare explicit baseline/candidate cases on the same snapshot to obtain paired
statistics; different datasets are descriptive only. The configuration forms expose
retrieval-only mode and the optional independent Ollama judge.
