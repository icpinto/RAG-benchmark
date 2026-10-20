import CodeMirror from "@uiw/react-codemirror";
import { yaml } from "@codemirror/lang-yaml";
export default function YamlEditor({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <CodeMirror
      value={value}
      height="650px"
      extensions={[yaml()]}
      onChange={onChange}
      aria-label="Benchmark configuration YAML"
      theme="light"
      basicSetup={{ lineNumbers: true, foldGutter: true }}
    />
  );
}
