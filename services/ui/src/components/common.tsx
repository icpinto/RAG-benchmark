import {
  CircleAlert,
  CircleHelp,
  CircleX,
  Clock,
  Info,
  TriangleAlert,
  Check,
  CheckCircle2,
  Inbox,
  LoaderCircle,
  Copy,
  Download,
  RotateCcw,
  type LucideIcon,
} from "lucide-react";
import type { ReactNode } from "react";
import { useState } from "react";
import { Button } from "./ui/button";
import { download } from "../lib/utils";
export function PageHeading({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="actions">{actions}</div>}
    </div>
  );
}
export function Card({
  title,
  description,
  children,
  className = "",
}: {
  title?: string;
  description?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {title && (
        <div className="card-heading">
          <h2>{title}</h2>
          {description && <p>{description}</p>}
        </div>
      )}
      {children}
    </section>
  );
}
export function Notice({
  children,
  tone = "info",
}: {
  children: ReactNode;
  tone?: "info" | "error" | "success" | "warning";
}) {
  const Icon = {
    info: Info,
    error: CircleAlert,
    warning: TriangleAlert,
    success: CheckCircle2,
  }[tone];
  return (
    <div
      className={`notice notice-${tone}`}
      role={tone === "error" ? "alert" : "status"}
    >
      <Icon aria-hidden="true" focusable="false" size={16} />
      <div>{children}</div>
    </div>
  );
}
export function ErrorNotice({
  error,
  retry,
}: {
  error: unknown;
  retry?: () => void;
}) {
  return error ? (
    <Notice tone="error">
      {error instanceof Error ? error.message : String(error)}
      {retry && (
        <Button variant="ghost" size="sm" onClick={retry}>
          <RotateCcw aria-hidden="true" focusable="false" size={14} />
          Try again
        </Button>
      )}
    </Notice>
  ) : null;
}
export function Loading({ label = "Loading workspace…" }: { label?: string }) {
  return (
    <div className="loading" role="status">
      <LoaderCircle
        aria-hidden="true"
        focusable="false"
        className="spin"
        size={20}
      />
      {label}
    </div>
  );
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty">
      <Inbox aria-hidden="true" focusable="false" size={32} />
      <h3>{title}</h3>
      {children && <p>{children}</p>}
    </div>
  );
}
export function StatusBadge({ status }: { status?: string }) {
  const normalized = status?.trim().toLowerCase() || "unknown";
  const icons: Record<string, LucideIcon> = {
    queued: Clock,
    running: LoaderCircle,
    completed: CheckCircle2,
    failed: CircleX,
  };
  const Icon = Object.hasOwn(icons, normalized)
    ? icons[normalized]
    : CircleHelp;
  return (
    <span className={`status-badge status-${normalized}`}>
      <Icon
        aria-hidden="true"
        focusable="false"
        size={14}
        className={normalized === "running" ? "spin" : undefined}
      />
      {normalized}
    </span>
  );
}
export function Tabs({
  items,
  value,
  onChange,
  label,
}: {
  items: { value: string; label: string; icon?: LucideIcon }[];
  value: string;
  onChange: (value: string) => void;
  label: string;
}) {
  return (
    <div className="tabs" role="group" aria-label={label}>
      {items.map((item) => (
        <button
          key={item.value}
          type="button"
          aria-pressed={value === item.value}
          className={value === item.value ? "tab active" : "tab"}
          onClick={() => onChange(item.value)}
        >
          {item.icon && (
            <item.icon aria-hidden="true" focusable="false" size={16} />
          )}
          {item.label}
        </button>
      ))}
    </div>
  );
}
export function JsonView({
  value,
  name = "data.json",
}: {
  value: unknown;
  name?: string;
}) {
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");
  const text = JSON.stringify(value, null, 2);
  return (
    <div>
      <div className="actions justify-end">
        <Button
          variant="ghost"
          size="sm"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(text);
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            } catch {
              setError("Clipboard unavailable. Use Download JSON.");
            }
          }}
        >
          {copied ? (
            <Check aria-hidden="true" focusable="false" size={14} />
          ) : (
            <Copy aria-hidden="true" focusable="false" size={14} />
          )}
          {copied ? "Copied" : "Copy JSON"}
        </Button>
        <Button variant="ghost" size="sm" onClick={() => download(name, text)}>
          <Download aria-hidden="true" focusable="false" size={14} />
          Download JSON
        </Button>
      </div>
      {error && <Notice tone="warning">{error}</Notice>}
      <pre className="code-panel">{text}</pre>
    </div>
  );
}
