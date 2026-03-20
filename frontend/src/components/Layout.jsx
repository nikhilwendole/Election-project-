import { useNavigate, useLocation } from "react-router-dom";

export default function Layout({ children }) {
  const navigate  = useNavigate();
  const location  = useLocation();

  const logout = () => {
    localStorage.removeItem("admin");
    navigate("/");
  };

  const navItems = [
    { path: "/dashboard", label: "📊 Dashboard" },
    { path: "/voters",    label: "👥 Voters"    },
    { path: "/reports",   label: "📄 Reports"   },
    { path: "/vote",      label: "🗳️ Cast Vote"  },
  ];

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      {/* Sidebar */}
      <div style={{
        width: "240px", background: "#1e293b",
        padding: "24px 0", display: "flex",
        flexDirection: "column", borderRight: "1px solid #334155"
      }}>
        <div style={{ padding: "0 24px 32px" }}>
          <div style={{ fontSize: "20px", fontWeight: "700", color: "#3b82f6" }}>
            🗳️ ElectionAI
          </div>
          <div style={{ fontSize: "12px", color: "#64748b", marginTop: "4px" }}>
            Face Detection System
          </div>
        </div>

        {navItems.map(item => (
          <div
            key={item.path}
            onClick={() => navigate(item.path)}
            style={{
              padding: "12px 24px", cursor: "pointer",
              background: location.pathname === item.path ? "#3b82f6" : "transparent",
              color: location.pathname === item.path ? "#fff" : "#94a3b8",
              fontSize: "14px", fontWeight: "500",
              borderLeft: location.pathname === item.path ? "3px solid #60a5fa" : "3px solid transparent",
              transition: "all 0.2s"
            }}
          >
            {item.label}
          </div>
        ))}

        <div style={{ marginTop: "auto", padding: "24px" }}>
          <button
            onClick={logout}
            style={{
              width: "100%", padding: "10px",
              background: "#ef4444", color: "#fff",
              border: "none", borderRadius: "8px",
              cursor: "pointer", fontSize: "14px"
            }}
          >
            🚪 Logout
          </button>
        </div>
      </div>

      {/* Main content */}
      <div style={{ flex: 1, padding: "32px", overflowY: "auto" }}>
        {children}
      </div>
    </div>
  );
}