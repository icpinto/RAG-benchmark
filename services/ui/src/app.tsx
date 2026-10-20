import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, NavLink, Outlet, useRouteError } from "react-router-dom";
import {
  Activity,
  BookOpen,
  ChartNoAxesCombined,
  ChevronRight,
  FlaskConical,
  Menu,
  Settings2,
  X,
} from "lucide-react";
import { DraftProvider } from "./hooks/draft";
import { api } from "./lib/api";
import { Button } from "./components/ui/button";
import { ErrorNotice, Loading } from "./components/common";
export const KnowledgeBase = lazy(() => import("./pages/knowledge-base"));
export const BenchmarkSetup = lazy(() => import("./pages/benchmark-setup"));
export const ResultsExplorer = lazy(() => import("./pages/results-explorer"));
const links = [
  {
    to: "/knowledge-base",
    label: "Knowledge Base",
    subtitle: "Documents & ground truth",
    icon: BookOpen,
  },
  {
    to: "/benchmark-setup",
    label: "Benchmark Setup",
    subtitle: "Pipelines & experiments",
    icon: Settings2,
  },
  {
    to: "/results",
    label: "Results Explorer",
    subtitle: "Metrics, comparisons & traces",
    icon: ChartNoAxesCombined,
  },
];
export function RouteError() {
  const error = useRouteError();
  return (
    <div className="error-page">
      <h1>Unable to open this workspace</h1>
      <ErrorNotice error={error} />
      <Button asChild>
        <Link to="/knowledge-base">Return to Knowledge Base</Link>
      </Button>
    </div>
  );
}
export default function App() {
  const [mobile, setMobile] = useState(false);
  const [narrow, setNarrow] = useState(
    () => window.matchMedia("(max-width: 900px)").matches,
  );
  const sidebar = useRef<HTMLElement>(null);
  useEffect(() => {
    const media = window.matchMedia("(max-width: 900px)");
    const update = () => {
      setNarrow(media.matches);
      if (!media.matches) setMobile(false);
    };
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    if (!mobile) return;
    const previous = document.activeElement;
    const focusable = () =>
      Array.from(
        sidebar.current?.querySelectorAll<HTMLElement>("a,button") || [],
      ).filter((element) => element.offsetParent !== null);
    sidebar.current
      ?.querySelector<HTMLElement>('[aria-label="Close navigation"]')
      ?.focus();
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobile(false);
      if (event.key === "Tab") {
        const entries = focusable();
        const first = entries[0];
        const last = entries.at(-1);
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", keyboard);
    return () => {
      document.removeEventListener("keydown", keyboard);
      if (previous instanceof HTMLElement) previous.focus();
    };
  }, [mobile]);
  const health = useQuery({
    queryKey: ["health"],
    queryFn: () => api("/health"),
    refetchInterval: 30000,
    retry: 1,
  });
  return (
    <DraftProvider>
      <div className="app-shell">
        {mobile && (
          <button
            className="sidebar-backdrop"
            aria-label="Close navigation"
            onClick={() => setMobile(false)}
          />
        )}
        <aside
          ref={sidebar}
          id="workspace-navigation"
          inert={narrow && !mobile}
          className={`sidebar ${mobile ? "open" : ""}`}
        >
          <Link
            className="brand"
            to="/knowledge-base"
            onClick={() => setMobile(false)}
          >
            <span className="brand-icon">
              <FlaskConical size={22} />
            </span>
            <span>
              RAG<span className="brand-light">bench</span>
              <small>EXPERIMENT WORKSPACE</small>
            </span>
          </Link>
          <Button
            variant="ghost"
            size="icon"
            className="mobile-close"
            aria-label="Close navigation"
            onClick={() => setMobile(false)}
          >
            <X size={20} />
          </Button>
          <div className="sidebar-label">YOUR WORKFLOW</div>
          <nav aria-label="Main navigation">
            {links.map(({ to, label, subtitle, icon: Icon }, index) => (
              <NavLink
                key={to}
                to={to}
                onClick={() => setMobile(false)}
                className={({ isActive }) =>
                  `nav-item ${isActive ? "active" : ""}`
                }
              >
                <Icon size={19} />
                <span>
                  {label}
                  <small>{subtitle}</small>
                </span>
                <span className="nav-step">0{index + 1}</span>
              </NavLink>
            ))}
          </nav>
          <div className="sidebar-note">
            <span className="eyebrow">Designed for discovery</span>
            <p>Prepare. Experiment. Understand.</p>
            <small>
              Compare RAG pipelines with reproducible configurations and
              inspectable evidence.
            </small>
          </div>
          <div className="sidebar-footer">
            <span
              className={`connection-dot ${health.isSuccess ? "online" : ""}`}
            />
            {health.isSuccess
              ? "API connected"
              : health.isPending
                ? "Connecting to API…"
                : "API unavailable"}
            <button
              className="text-link"
              aria-label="Retry API connection"
              onClick={() => {
                void health.refetch();
              }}
            >
              <Activity size={14} />
            </button>
          </div>
        </aside>
        <div className="workspace">
          <header className="topbar">
            <div className="actions">
              <Button
                variant="ghost"
                size="icon"
                className="mobile-menu"
                aria-label="Open navigation"
                aria-controls="workspace-navigation"
                aria-expanded={mobile}
                onClick={() => setMobile(true)}
              >
                <Menu size={20} />
              </Button>
              <span className="topbar-label">Workspace</span>
              <ChevronRight size={14} />
              <strong>RAG Benchmark Platform</strong>
            </div>
            <a
              href={
                import.meta.env.VITE_API_DOCS_URL ||
                `${window.location.protocol}//${window.location.hostname}:7800/docs`
              }
              target="_blank"
              rel="noreferrer"
              className="docs-link"
            >
              API documentation <span>↗</span>
            </a>
          </header>
          <main id="main-content">
            <Suspense fallback={<Loading />}>
              <Outlet />
            </Suspense>
          </main>
          <footer className="workspace-footer">
            RAG Benchmark Platform
            <span>Make every experiment explainable.</span>
          </footer>
        </div>
      </div>
    </DraftProvider>
  );
}
