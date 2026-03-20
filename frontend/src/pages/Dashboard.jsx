import { useState, useEffect } from "react";
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function Dashboard() {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    axios.get("http://localhost:5000/api/stats")
      .then(r => setStats(r.data))
      .catch(() => {});
  }, []);

  const cards = [
    { label: "Total Voters",  value: stats?.total_voters ?? "...", icon: "👥", color: "#3b82f6" },
    { label: "Voted Today",   value: stats?.today_votes  ?? "...", icon: "✅", color: "#10b981" },
    { label: "Duplicate Blocked", value: "0", icon: "🚫", color: "#ef4444" },
  ];

  return (
    <div>
      <h1 style={{ fontSize: "28px", fontWeight: "700", marginBottom: "8px" }}>
        Dashboard
      </h1>
      <p style={{ color: "#64748b", marginBottom: "32px" }}>
        Election overview and statistics
      </p>

      {/* Stat Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "20px", marginBottom: "32px" }}>
        {cards.map((card, i) => (
          <div key={i} style={{
            background: "#1e293b", borderRadius: "12px",
            padding: "24px", borderLeft: `4px solid ${card.color}`
          }}>
            <div style={{ fontSize: "32px", marginBottom: "8px" }}>{card.icon}</div>
            <div style={{ fontSize: "36px", fontWeight: "700", color: card.color }}>
              {card.value}
            </div>
            <div style={{ color: "#64748b", fontSize: "14px", marginTop: "4px" }}>
              {card.label}
            </div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div style={{ background: "#1e293b", borderRadius: "12px", padding: "24px" }}>
        <h2 style={{ fontSize: "18px", fontWeight: "600", marginBottom: "20px" }}>
          📊 Votes by Hour
        </h2>
        {stats?.hourly_data?.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={stats.hourly_data}>
              <XAxis dataKey="hour" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
              />
              <Bar dataKey="votes" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ textAlign: "center", color: "#64748b", padding: "60px" }}>
            No voting data yet
          </div>
        )}
      </div>
    </div>
  );
}