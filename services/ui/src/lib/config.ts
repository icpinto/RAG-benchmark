import {
  isAlias,
  isMap,
  isNode,
  isSeq,
  parseDocument,
  type Document,
  type Node,
} from "yaml";
import { isRecord } from "./utils";
export type ConfigPath = (string | number)[];
export function parseConfig(text: string) {
  const doc: Document = parseDocument(text, {
    version: "1.1",
    uniqueKeys: false,
    keepSourceTokens: true,
  });
  const errors = doc.errors.map((error) => error.message);
  if (!errors.length && doc.contents !== null && !isMap(doc.contents))
    errors.push("Config YAML must contain a mapping at the document root.");
  if (!errors.length) {
    try {
      doc.toJS();
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }
  // Duplicate keys remain in the AST; Python safe_load uses the last occurrence.
  const duplicates: string[] = [];
  function visit(node: unknown, path: string) {
    if (isMap(node)) {
      const seen = new Set<string>();
      for (const pair of node.items) {
        const key = String(pair.key);
        if (seen.has(key)) duplicates.push(path + key);
        seen.add(key);
        visit(pair.value, `${path}${key}.`);
      }
    } else if (isSeq(node))
      node.items.forEach((item, i) => visit(item, `${path}${i}.`));
  }
  visit(doc.contents, "");
  return { doc, errors, duplicates };
}
function child(doc: Document, node: unknown, key: string | number): unknown {
  if (isAlias(node)) node = node.resolve(doc);
  if (isMap(node))
    return [...node.items]
      .reverse()
      .find((pair) => String(pair.key) === String(key))?.value;
  if (isSeq(node)) return node.items[Number(key)];
  return undefined;
}
export function configNode(doc: Document, path: ConfigPath): unknown {
  let node: unknown = doc.contents;
  for (const key of path) node = child(doc, node, key);
  return node;
}
const values = new WeakMap<Document, unknown>();
export function configValue(doc: Document, path: ConfigPath): unknown {
  if (!values.has(doc)) values.set(doc, doc.toJS());
  let value = values.get(doc);
  for (const key of path)
    value =
      value !== null && typeof value === "object"
        ? (value as Record<string | number, unknown>)[key]
        : undefined;
  return value;
}
function replacement(doc: Document, previous: unknown, value: unknown): Node {
  // Reuse unchanged nodes when a form adds/removes list entries, retaining
  // comments, styles and anchors on the other component variants.
  if (
    isNode(previous) &&
    JSON.stringify(previous.toJS(doc)) === JSON.stringify(value)
  )
    return previous;
  if (isMap(previous) && isRecord(value)) {
    for (const key of new Set(previous.items.map((pair) => String(pair.key)))) {
      const pair = [...previous.items]
        .reverse()
        .find((pair) => String(pair.key) === key)!;
      if (Object.hasOwn(value, key))
        pair.value = replacement(doc, pair.value, value[key]);
      else
        previous.items = previous.items.filter(
          (pair) => String(pair.key) !== key,
        );
    }
    for (const [key, child] of Object.entries(value))
      if (!previous.items.some((pair) => String(pair.key) === key))
        previous.add(doc.createPair(key, child));
    return previous;
  }
  if (isSeq(previous) && Array.isArray(value)) {
    const original = [...previous.items];
    const used = new Set<number>();
    previous.items = value.map((next, index) => {
      const match = original.findIndex(
        (node, candidate) =>
          !used.has(candidate) &&
          isNode(node) &&
          JSON.stringify(node.toJS(doc)) === JSON.stringify(next),
      );
      if (match >= 0) {
        used.add(match);
        return original[match];
      }
      return replacement(
        doc,
        used.has(index) ? undefined : original[index],
        next,
      );
    });
    return previous;
  }
  const next = doc.createNode(value);
  if (isNode(previous)) {
    next.comment = previous.comment;
    next.commentBefore = previous.commentBefore;
    if ("anchor" in previous && "anchor" in next) next.anchor = previous.anchor;
  }
  return next;
}
export function updateConfig(
  text: string,
  path: ConfigPath,
  value: unknown,
): string {
  const { doc, errors } = parseConfig(text);
  if (errors.length) throw new Error(errors[0]);
  if (!doc.contents) doc.contents = doc.createNode({});
  let node: unknown = doc.contents;
  for (let i = 0; i < path.length; i++) {
    const key = path[i];
    const last = i === path.length - 1;
    if (isMap(node)) {
      let pair = [...node.items]
        .reverse()
        .find((item) => String(item.key) === String(key));
      if (last) {
        if (value === undefined) {
          if (pair) node.items.splice(node.items.indexOf(pair), 1);
        } else if (pair) pair.value = replacement(doc, pair.value, value);
        else node.add(doc.createPair(key, value));
      } else {
        if (!pair) {
          pair = doc.createPair(key, typeof path[i + 1] === "number" ? [] : {});
          node.add(pair);
        }
        if (isAlias(pair.value)) {
          const resolved = pair.value.resolve(doc);
          pair.value = resolved?.clone() ?? doc.createNode({});
          if (
            pair.value &&
            typeof pair.value === "object" &&
            "anchor" in pair.value
          )
            pair.value.anchor = undefined;
        }
        node = pair.value;
      }
    } else if (isSeq(node)) {
      const index = Number(key);
      if (last) {
        if (value === undefined) node.items.splice(index, 1);
        else node.items[index] = replacement(doc, node.items[index], value);
      } else {
        if (!node.items[index])
          node.items[index] = doc.createNode(
            typeof path[i + 1] === "number" ? [] : {},
          );
        if (isAlias(node.items[index])) {
          const cloned =
            node.items[index].resolve(doc)?.clone() ?? doc.createNode({});
          if ("anchor" in cloned) cloned.anchor = undefined;
          node.items[index] = cloned;
        }
        node = node.items[index];
      }
    } else
      throw new Error(
        "This YAML structure cannot be edited with forms. Adjust it in the YAML editor.",
      );
  }
  return doc.toString({ lineWidth: 0 });
}
