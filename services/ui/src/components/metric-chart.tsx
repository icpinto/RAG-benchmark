import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
export default function MetricChart({
  data,
}: {
  data: { name: string; value: number }[];
}) {
  return (
    <div className="chart" role="img" aria-label="Metric values bar chart">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          margin={{ top: 16, right: 16, bottom: 40, left: 0 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            vertical={false}
            stroke="#e2e8f0"
          />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 14, fill: "#475569" }}
            interval={0}
            angle={-12}
            textAnchor="end"
          />
          <YAxis tick={{ fontSize: 14, fill: "#475569" }} />
          <Tooltip contentStyle={{ color: "#0f172a", fontSize: 14 }} />
          <Bar
            dataKey="value"
            fill="#1d4ed8"
            radius={[5, 5, 0, 0]}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
