import { useState, useEffect } from "react";
import axios from "axios";

export default function Voters() {
  const [voters,  setVoters]  = useState([]);
  const [search,  setSearch]  = useState("");
  const [loading, setLoading] = useState(true);

  const fetchVoters = () => {
    setLoading(true);
    axios.get("http://localhost:5000/api/voters")
      .then(r => { setVoters(r.data.voters); setLoading(false); })
      .catch(() => setLoading(false));
  };

  useEffect(() => { fetchVoters(); }, []);

  const deleteVoter = async (voter_id, name) => {
    if (!window.confirm(`Delete voter "${name}"?`)) return;
    await axios.delete(`http://localhost:5000/api/voters/${voter_id}`);
    fetchVoters();
  };

  const filtered = voters.filter(v =>
    v.voter_name?.toLowerCase().includes(search.toLowerCase()) ||
    v.voter_number?.toLowerCase().includes(search.toLowerCase()) ||
    v.voter_id?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <h1 style={{ fontSize: "28px", fontWeight: "700", marginBottom: "8px" }}>
        Voters
      </h1>
      <p style={{ color: "#64748b", marginBottom: "32px" }}>
        All registered voters in MongoDB
      </p>

      {/* Search */}
      <input
        value={search}
        onChange={e => setSearch(e.target.value)}
        placeholder="🔍 Search by name, voter number or ID..."
        style={{
          width: "100%", padding: "12px 16px", marginBottom: "20px",
          background: "#1e293b", border: "1px solid #334155",
          borderRadius: "8px", color: "#fff", fontSize: "14px", outline: "none"
        }}
      />

      {/* Table */}
      <div style={{ background: "#1e293b", borderRadius: "12px", overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "#0f172a" }}>
              {["#", "Name", "Voter Number", "Voter ID", "Voted At", "Action"].map(h => (
                <th key={h} style={{
                  padding: "14px 16px", textAlign: "left",
                  color: "#64748b", fontSize: "12px",
                  fontWeight: "600", textTransform: "uppercase"
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan="6" style={{ textAlign: "center", padding: "40px", color: "#64748b" }}>
                  Loading...
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan="6" style={{ textAlign: "center", padding: "40px", color: "#64748b" }}>
                  No voters found
                </td>
              </tr>
            ) : filtered.map((v, i) => (
              <tr key={v.voter_id} style={{
                borderTop: "1px solid #334155",
                background: i % 2 === 0 ? "transparent" : "#0f172a22"
              }}>
                <td style={{ padding: "14px 16px", color: "#64748b", fontSize: "13px" }}>
                  {i + 1}
                </td>
                <td style={{ padding: "14px 16px", fontWeight: "600" }}>
                  {v.voter_name}
                </td>
                <td style={{ padding: "14px 16px", color: "#94a3b8" }}>
                  {v.voter_number}
                </td>
                <td style={{ padding: "14px 16px", fontFamily: "monospace", fontSize: "12px", color: "#60a5fa" }}>
                  {v.voter_id}
                </td>
                <td style={{ padding: "14px 16px", color: "#94a3b8", fontSize: "13px" }}>
                  {v.voted_at?.slice(0, 16)}
                </td>
                <td style={{ padding: "14px 16px" }}>
                  <button
                    onClick={() => deleteVoter(v.voter_id, v.voter_name)}
                    style={{
                      padding: "6px 12px", background: "#450a0a",
                      color: "#ef4444", border: "1px solid #ef4444",
                      borderRadius: "6px", cursor: "pointer", fontSize: "12px"
                    }}
                  >
                    🗑️ Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: "12px", color: "#64748b", fontSize: "13px" }}>
        Total: {filtered.length} voters
      </div>
    </div>
  );
}