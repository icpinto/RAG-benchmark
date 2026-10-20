import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createBrowserRouter,
  Navigate,
  RouterProvider,
} from "react-router-dom";
import App, {
  BenchmarkSetup,
  KnowledgeBase,
  ResultsExplorer,
  RouteError,
} from "./app";
import "./styles.css";
const client = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 1000, refetchOnWindowFocus: true },
    mutations: { retry: false },
  },
});
const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Navigate to="/knowledge-base" replace /> },
      { path: "knowledge-base", element: <KnowledgeBase /> },
      { path: "benchmark-setup", element: <BenchmarkSetup /> },
      { path: "results", element: <ResultsExplorer /> },
      { path: "results/:runId", element: <ResultsExplorer /> },
      { path: "*", element: <Navigate to="/knowledge-base" replace /> },
    ],
  },
]);
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>,
);
