import { useEffect, useId, useState } from "react";
import type { Document } from "yaml";
import { parse } from "yaml";
import { Plus, RotateCcw, Trash2, Layers, Variable, ListX } from "lucide-react";
import type { Field, Stage } from "../lib/types";
import { configValue, type ConfigPath } from "../lib/config";
import { isRecord } from "../lib/utils";
import { Button } from "./ui/button";
import { ErrorNotice, Notice } from "./common";

type Change = (path: ConfigPath, value: unknown) => void;
function ValueInput({
  value,
  onChange,
  label,
}: {
  value: unknown;
  onChange: (value: unknown) => void;
  label: string;
}) {
  const serialized = value === undefined ? "" : JSON.stringify(value, null, 2);
  const [text, setText] = useState(serialized);
  const [error, setError] = useState("");
  useEffect(() => {
    setText(serialized);
    setError("");
  }, [serialized]);
  return (
    <div>
      <textarea
        aria-label={label}
        rows={3}
        value={text}
        onChange={(event) => {
          setText(event.target.value);
          try {
            const parsed: unknown = parse(event.target.value, {
              version: "1.1",
              uniqueKeys: false,
            });
            onChange(parsed);
            setError("");
          } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
          }
        }}
      />
      {error && (
        <span className="field-error" role="alert">
          {error}
        </span>
      )}
      <span className="field-help">
        YAML value: scalar, array, or nested mapping.
      </span>
    </div>
  );
}
export function FieldInput({
  field,
  value,
  onChange,
}: {
  field: Field;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const id = useId();
  const [expression, setExpression] = useState(false);
  const complex =
    field.type === "value" || Array.isArray(value) || isRecord(value);
  const env = expression || (typeof value === "string" && value.includes("${"));
  let control;
  if (complex)
    control = (
      <ValueInput
        key={id}
        label={field.label}
        value={value}
        onChange={onChange}
      />
    );
  else if (field.type === "boolean" && !env)
    control = (
      <select
        id={id}
        value={value === undefined ? "" : String(value)}
        onChange={(event) =>
          onChange(
            event.target.value === ""
              ? undefined
              : event.target.value === "true",
          )
        }
      >
        <option value="">Default ({String(field.default ?? false)})</option>
        <option value="true">Enabled</option>
        <option value="false">Disabled</option>
      </select>
    );
  else if (
    field.choices &&
    !env &&
    (value === undefined || field.choices.includes(String(value)))
  )
    control = (
      <select
        id={id}
        value={value === undefined ? "" : String(value)}
        onChange={(event) => onChange(event.target.value || undefined)}
      >
        <option value="">
          {field.default === undefined
            ? "Choose…"
            : `Default (${String(field.default)})`}
        </option>
        {field.choices.map((choice) => (
          <option key={choice}>{choice}</option>
        ))}
      </select>
    );
  else
    control = (
      <input
        id={id}
        type={
          field.secret
            ? "password"
            : ["number", "integer"].includes(field.type) && !env
              ? "number"
              : "text"
        }
        step={field.type === "integer" ? "1" : "any"}
        min={field.minimum}
        max={field.maximum}
        placeholder={
          field.default === undefined ? "Not set" : String(field.default)
        }
        value={value === undefined || value === null ? "" : String(value)}
        onChange={(event) => {
          const input = event.target.value;
          onChange(
            input === ""
              ? undefined
              : ["number", "integer"].includes(field.type) && !env
                ? Number(input)
                : input,
          );
        }}
      />
    );
  return (
    <div className="field">
      <div className="field-label">
        <label htmlFor={complex ? undefined : id}>
          {field.label}
          {field.required && <span className="required"> *</span>}
        </label>
        <span className="actions">
          {field.sweepable && (
            <Button
              variant="ghost"
              size="sm"
              aria-label={`Toggle sweep for ${field.label}`}
              onClick={() =>
                onChange(
                  Array.isArray(value)
                    ? value[0]
                    : [value ?? field.default ?? 0],
                )
              }
            >
              <Layers aria-hidden="true" focusable="false" size={14} />
              Sweep
            </Button>
          )}
          {field.type !== "value" && (
            <Button
              variant="ghost"
              size="sm"
              aria-label={`Use expression for ${field.label}`}
              onClick={() => {
                setExpression(!env);
                if (env) onChange(undefined);
              }}
            >
              <Variable aria-hidden="true" focusable="false" size={14} />
              Expression
            </Button>
          )}
          {value !== undefined && (
            <Button
              variant="ghost"
              size="icon"
              aria-label={`Reset ${field.label}`}
              onClick={() => onChange(undefined)}
            >
              <RotateCcw aria-hidden="true" focusable="false" size={14} />
            </Button>
          )}
        </span>
      </div>
      {control}
      {field.help && <span className="field-help">{field.help}</span>}
    </div>
  );
}
export function Fields({
  fields,
  doc,
  path,
  onChange,
}: {
  fields: Field[];
  doc: Document;
  path: ConfigPath;
  onChange: Change;
}) {
  return (
    <div className="form-grid">
      {fields.map((field) => {
        const keys = [field.key, ...(field.aliases || [])];
        const actual =
          keys.find((key) => configValue(doc, [...path, key]) !== undefined) ??
          field.key;
        const target = [...path, actual];
        const value = configValue(doc, target);
        if (
          field.type === "object" &&
          (!field.sweepable || !Array.isArray(value))
        ) {
          if (value !== undefined && !isRecord(value))
            return (
              <FieldInput
                key={actual}
                field={{ ...field, type: "value" }}
                value={value}
                onChange={(next) => onChange(target, next)}
              />
            );
          return (
            <details className="nested-fields" key={actual}>
              <summary>
                {field.label}
                <span>{value === undefined ? "Default" : "Configured"}</span>
              </summary>
              {field.help && <p className="field-help">{field.help}</p>}
              {field.sweepable && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onChange(target, [value ?? {}])}
                >
                  <Layers aria-hidden="true" focusable="false" size={14} />
                  Use object sweep
                </Button>
              )}
              {field.fields?.length ? (
                <Fields
                  fields={field.fields}
                  doc={doc}
                  path={target}
                  onChange={onChange}
                />
              ) : (
                <ObjectEditor
                  value={isRecord(value) ? value : {}}
                  onChange={(next) => onChange(target, next)}
                />
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onChange(target, undefined)}
              >
                <RotateCcw aria-hidden="true" focusable="false" size={14} />
                Reset {field.label}
              </Button>
            </details>
          );
        }
        if (field.type === "object" && field.sweepable && Array.isArray(value))
          return (
            <div className="nested-fields" key={actual}>
              <div className="field-label">
                <span>{field.label} sweep</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onChange(target, value[0])}
                >
                  <Layers aria-hidden="true" focusable="false" size={14} />
                  Use single value
                </Button>
              </div>
              <ArrayEditor
                values={value}
                item={{ ...field, sweepable: false }}
                onChange={(next) => onChange(target, next)}
              />
            </div>
          );
        if (
          field.type === "array" &&
          (value === undefined || Array.isArray(value))
        )
          return (
            <div className="nested-fields" key={actual}>
              <div className="field-label">
                <label>{field.label}</label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onChange(target, undefined)}
                >
                  <RotateCcw aria-hidden="true" focusable="false" size={14} />
                  Use default
                </Button>
              </div>
              {field.help && <p className="field-help">{field.help}</p>}
              <ArrayEditor
                values={value as unknown[] | undefined}
                item={
                  field.items || { key: "item", label: "Item", type: "value" }
                }
                onChange={(next) => onChange(target, next)}
              />
              {value === undefined && field.default !== undefined && (
                <span className="field-help">
                  Default: {JSON.stringify(field.default)}
                </span>
              )}
            </div>
          );
        return (
          <FieldInput
            key={actual}
            field={field}
            value={value}
            onChange={(next) => onChange(target, next)}
          />
        );
      })}
    </div>
  );
}
function ArrayEditor({
  values,
  item,
  onChange,
}: {
  values?: unknown[];
  item: Field;
  onChange: (value: unknown[]) => void;
}) {
  return (
    <div className="stack">
      {(values || []).map((value, index) => (
        <div className="array-item" key={index}>
          {item.type === "object" && isRecord(value) ? (
            <ObjectEditor
              value={value}
              fields={item.fields}
              onChange={(next) =>
                onChange(values!.map((v, i) => (i === index ? next : v)))
              }
            />
          ) : (
            <NestedValue
              field={{ ...item, label: `Item ${index + 1}` }}
              value={value}
              onChange={(next) =>
                onChange(values!.map((v, i) => (i === index ? next : v)))
              }
            />
          )}
          <Button
            variant="ghost"
            size="icon"
            aria-label={`Remove item ${index + 1}`}
            onClick={() => onChange(values!.filter((_, i) => i !== index))}
          >
            <Trash2 aria-hidden="true" focusable="false" size={14} />
          </Button>
        </div>
      ))}
      <div className="actions">
        <Button
          variant="outline"
          size="sm"
          onClick={() =>
            onChange([
              ...(values || []),
              item.type === "object"
                ? {}
                : item.type === "integer" || item.type === "number"
                  ? 1
                  : "",
            ])
          }
        >
          <Plus aria-hidden="true" focusable="false" size={14} />
          Add item
        </Button>
        {values === undefined && (
          <Button variant="ghost" size="sm" onClick={() => onChange([])}>
            <ListX aria-hidden="true" focusable="false" size={14} />
            Set empty list
          </Button>
        )}
      </div>
    </div>
  );
}
function NestedValue({
  field,
  value,
  onChange,
}: {
  field: Field;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const [raw, setRaw] = useState(false);
  const kind = raw
    ? "yaml"
    : isRecord(value)
      ? "object"
      : Array.isArray(value)
        ? "array"
        : value === null
          ? "null"
          : typeof value === "number"
            ? "number"
            : typeof value === "boolean"
              ? "boolean"
              : "string";
  if (field.type !== "value" && !["object", "array"].includes(field.type))
    return <FieldInput field={field} value={value} onChange={onChange} />;
  return (
    <fieldset className="nested-fields">
      <legend>{field.label}</legend>
      <label className="field">
        Value type
        <select
          aria-label={`Value type for ${field.label}`}
          value={kind}
          onChange={(event) => {
            const next = event.target.value;
            setRaw(next === "yaml");
            if (next !== "yaml")
              onChange(
                next === "object"
                  ? {}
                  : next === "array"
                    ? []
                    : next === "number"
                      ? 0
                      : next === "boolean"
                        ? false
                        : next === "null"
                          ? null
                          : "",
              );
          }}
        >
          <option value="string">Text / expression</option>
          <option value="number">Number</option>
          <option value="boolean">Boolean</option>
          <option value="object">Nested mapping</option>
          <option value="array">List</option>
          <option value="null">Null</option>
          <option value="yaml">YAML value</option>
        </select>
      </label>
      {kind === "object" ? (
        <ObjectEditor
          value={value as Record<string, unknown>}
          fields={field.fields}
          onChange={onChange}
        />
      ) : kind === "array" ? (
        <ArrayEditor
          values={value as unknown[]}
          item={field.items || { key: "item", label: "Item", type: "value" }}
          onChange={onChange}
        />
      ) : kind === "null" ? (
        <span className="field-help">Explicit null</span>
      ) : kind === "yaml" ? (
        <ValueInput label={field.label} value={value} onChange={onChange} />
      ) : (
        <FieldInput
          field={{ ...field, type: kind }}
          value={value}
          onChange={onChange}
        />
      )}
    </fieldset>
  );
}
function ObjectEditor({
  value,
  fields = [],
  onChange,
}: {
  value: Record<string, unknown>;
  fields?: Field[];
  onChange: (value: Record<string, unknown>) => void;
}) {
  const [key, setKey] = useState("");
  const known = new Set(fields.map((field) => field.key));
  const all = [
    ...fields,
    ...Object.keys(value)
      .filter((k) => !known.has(k))
      .map((k) => ({ key: k, label: k, type: "value" as const })),
  ];
  return (
    <div className="stack">
      {all.map((field) => (
        <div className="array-item" key={field.key}>
          <NestedValue
            field={field}
            value={value[field.key]}
            onChange={(next) => {
              const copy = { ...value };
              if (next === undefined) delete copy[field.key];
              else copy[field.key] = next;
              onChange(copy);
            }}
          />
          <Button
            variant="ghost"
            size="icon"
            aria-label={`Remove ${field.label}`}
            onClick={() => {
              const copy = { ...value };
              delete copy[field.key];
              onChange(copy);
            }}
          >
            <Trash2 aria-hidden="true" focusable="false" size={14} />
          </Button>
        </div>
      ))}
      <div className="actions">
        <input
          aria-label="New parameter name"
          placeholder="Parameter name"
          value={key}
          onChange={(event) => setKey(event.target.value)}
        />
        <Button
          variant="outline"
          size="sm"
          disabled={!key.trim() || Object.hasOwn(value, key.trim())}
          onClick={() => {
            onChange({ ...value, [key.trim()]: "" });
            setKey("");
          }}
        >
          <Plus aria-hidden="true" focusable="false" size={14} />
          Add parameter
        </Button>
      </div>
    </div>
  );
}
export function StageEditor({
  stage,
  doc,
  onChange,
}: {
  stage: Stage;
  doc: Document;
  onChange: Change;
}) {
  const [error, setError] = useState("");
  const current = configValue(doc, stage.path);
  const list =
    stage.mode === "single"
      ? current === undefined
        ? []
        : [current]
      : stage.mode === "indexer"
        ? current === undefined
          ? []
          : Array.isArray(current)
            ? current
            : [current]
        : Array.isArray(current)
          ? current
          : [];
  const change: Change = (path, value) => {
    try {
      onChange(path, value);
      setError("");
    } catch (err) {
      setError(String(err));
    }
  };
  function setList(next: unknown[]) {
    change(
      stage.path,
      stage.mode === "single"
        ? next[0]
        : stage.mode === "indexer" && next.length <= 1
          ? next[0]
          : next,
    );
  }
  return (
    <div className="stack">
      <p className="field-help">
        {stage.help ||
          "Add component variants to include them in the benchmark configuration."}
      </p>
      <ErrorNotice error={error} />
      {current !== undefined &&
        !Array.isArray(current) &&
        !["single", "indexer"].includes(stage.mode) && (
          <Notice tone="warning">
            This stage has a custom YAML shape. Edit it in YAML to preserve it.
          </Notice>
        )}
      {list.map((value, index) => {
        const componentName =
          stage.mode === "extractors"
            ? String(value)
            : isRecord(value)
              ? String(value[stage.discriminator] || "")
              : "";
        const definition = stage.components.find(
          (component) => component.name === componentName,
        );
        const path: ConfigPath =
          stage.mode === "extractors"
            ? ["pipeline", "ingestion", "extractor_configs", componentName]
            : stage.mode === "single" ||
                (stage.mode === "indexer" && !Array.isArray(current))
              ? stage.path
              : [...stage.path, index];
        return (
          <div className="component-card" key={`${index}-${componentName}`}>
            <div className="component-title">
              <div>
                <span className="eyebrow">Variant {index + 1}</span>
                <h3>{componentName || "Custom component"}</h3>
              </div>
              <Button
                variant="ghost"
                size="icon"
                aria-label={`Remove ${componentName} variant ${index + 1}`}
                onClick={() => setList(list.filter((_, i) => i !== index))}
              >
                <Trash2 aria-hidden="true" focusable="false" size={14} />
              </Button>
            </div>
            {definition ? (
              <Fields
                fields={definition.fields}
                doc={doc}
                path={path}
                onChange={change}
              />
            ) : (
              <Notice>
                Custom component parameters remain available in YAML.
              </Notice>
            )}
            {definition?.help && (
              <p className="field-help">{definition.help}</p>
            )}
          </div>
        );
      })}
      <div className="actions">
        <select
          aria-label={`Add ${stage.label} component`}
          value=""
          onChange={(event) => {
            if (event.target.value)
              setList([
                ...list,
                stage.mode === "extractors"
                  ? event.target.value
                  : { [stage.discriminator]: event.target.value },
              ]);
          }}
          disabled={stage.mode === "single" && list.length > 0}
        >
          <option value="">Add {stage.label.toLowerCase()} component</option>
          {stage.components.map((component) => (
            <option key={component.name} value={component.name}>
              {component.name}
            </option>
          ))}
        </select>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => change(stage.path, undefined)}
        >
          <RotateCcw aria-hidden="true" focusable="false" size={14} />
          Use engine default
        </Button>
        {stage.mode === "variants" && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => change(stage.path, [])}
          >
            <ListX aria-hidden="true" focusable="false" size={14} />
            Set empty list
          </Button>
        )}
      </div>
    </div>
  );
}
