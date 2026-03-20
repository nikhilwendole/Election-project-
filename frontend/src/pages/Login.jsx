import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error,    setError]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const navigate = useNavigate();

  const handleLogin = async () => {
    if (!username || !password) {
      setError("Please enter username and password");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await axios.post("http://localhost:5000/api/login", {
        username, password
      });
      if (res.data.success) {
        localStorage.setItem("admin", username);
        navigate("/dashboard");
      }
    } catch {
      setError("Invalid username or password");
    }
    setLoading(false);
  };

  return (
    <div style={{
      minHeight: "100vh", display: "flex",
      alignItems: "center", justifyContent: "center",
      background: "linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%)"
    }}>
      <div style={{
        background: "#1e293b", padding: "48px",
        borderRadius: "16px", width: "400px",
        boxShadow: "0 25px 50px rgba(0,0,0,0.5)"
      }}>
        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: "32px" }}>
          <div style={{ fontSize: "48px" }}>🗳️</div>
          <h1 style={{ fontSize: "24px", fontWeight: "700", color: "#fff", marginTop: "8px" }}>
            ElectionAI
          </h1>
          <p style={{ color: "#64748b", fontSize: "14px", marginTop: "4px" }}>
            Face Detection Voting System
          </p>
        </div>

        {/* Form */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div>
            <label style={{ fontSize: "13px", color: "#94a3b8", marginBottom: "6px", display: "block" }}>
              Username
            </label>
            <input
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="admin"
              style={{
                width: "100%", padding: "12px 16px",
                background: "#0f172a", border: "1px solid #334155",
                borderRadius: "8px", color: "#fff", fontSize: "14px",
                outline: "none"
              }}
            />
          </div>

          <div>
            <label style={{ fontSize: "13px", color: "#94a3b8", marginBottom: "6px", display: "block" }}>
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              onKeyDown={e => e.key === "Enter" && handleLogin()}
              style={{
                width: "100%", padding: "12px 16px",
                background: "#0f172a", border: "1px solid #334155",
                borderRadius: "8px", color: "#fff", fontSize: "14px",
                outline: "none"
              }}
            />
          </div>

          {error && (
            <div style={{
              background: "#450a0a", border: "1px solid #ef4444",
              borderRadius: "8px", padding: "10px 16px",
              color: "#fca5a5", fontSize: "13px"
            }}>
              ❌ {error}
            </div>
          )}

          <button
            onClick={handleLogin}
            disabled={loading}
            style={{
              padding: "13px", background: "#3b82f6",
              color: "#fff", border: "none", borderRadius: "8px",
              fontSize: "15px", fontWeight: "600",
              cursor: loading ? "not-allowed" : "pointer",
              opacity: loading ? 0.7 : 1, marginTop: "8px"
            }}
          >
            {loading ? "Logging in..." : "Login →"}
          </button>

          <button
            onClick={() => navigate("/vote")}
            style={{
              padding: "13px", background: "transparent",
              color: "#3b82f6", border: "1px solid #3b82f6",
              borderRadius: "8px", fontSize: "15px",
              fontWeight: "600", cursor: "pointer"
            }}
          >
            🗳️ Go to Voting Booth
          </button>
        </div>
      </div>
    </div>
  );
}