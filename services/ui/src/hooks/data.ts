import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { DataFile, Question } from "../lib/types";
export const useFiles = () =>
  useQuery({
    queryKey: ["files"],
    queryFn: () => api<{ data_root: string; files: DataFile[] }>("/data/files"),
  });
export const useQuestions = () =>
  useQuery({
    queryKey: ["questions"],
    queryFn: () =>
      api<{ path: string; questions: Question[] }>("/data/questions"),
  });
