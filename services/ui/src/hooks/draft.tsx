import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
interface Draft {
  text: string;
  saved: string;
  path: string;
  loaded: boolean;
}
const initial: Draft = {
  text: "",
  saved: "",
  path: "/app/configs/default.yaml",
  loaded: false,
};
const Context = createContext<{
  draft: Draft;
  setDraft: React.Dispatch<React.SetStateAction<Draft>>;
} | null>(null);
export function DraftProvider({ children }: { children: ReactNode }) {
  const [draft, setDraft] = useState(initial);
  useEffect(() => {
    const listener = (event: BeforeUnloadEvent) => {
      if (draft.text !== draft.saved) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", listener);
    return () => window.removeEventListener("beforeunload", listener);
  }, [draft.text, draft.saved]);
  return (
    <Context.Provider value={{ draft, setDraft }}>{children}</Context.Provider>
  );
}
export function useDraft() {
  const context = useContext(Context);
  if (!context) throw new Error("Draft provider is missing");
  return context;
}
