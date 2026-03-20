import axios from "axios";

export default function Reports() {
  const downloadExcel = () => {
    window.open("http://localhost:5000/api/export/excel", "_blank");
  };

  const downloadPDF = () => {
    window.open("http://localhost:5000/api/export/pdf", "_blank");
  };

  return (
    <div>
      <h1 style={{ fontSize: "28px", fontWeight: "700", marginBottom: "8px" }}>
        Reports
      </h1>
      <p style={{ color: "#64748b", marginBottom: "32px" }}>
        Export voter data in different formats
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "20px" }}>
        {/* Excel */}
        <div style={{
          background: "#1e293b", borderRadius: "12px",
          padding: "32px", textAlign: "center",
          border: "1px solid #334155"
        }}>
          <div style={{ fontSize: "64px", marginBottom: "16px" }}>📊</div>
          <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "8px" }}>
            Excel Report
          </h2>
          <p style={{ color: "#64748b", fontSize: "14px", marginBottom: "24px" }}>
            Download all voters as .xlsx file with formatted table
          </p>
          <button
            onClick={downloadExcel}
            style={{
              padding: "12px 32px", background: "#16a34a",
              color: "#fff", border: "none", borderRadius: "8px",
              fontSize: "15px", fontWeight: "600", cursor: "pointer"
            }}
          >
            ⬇️ Download Excel
          </button>
        </div>

        {/* PDF */}
        <div style={{
          background: "#1e293b", borderRadius: "12px",
          padding: "32px", textAlign: "center",
          border: "1px solid #334155"
        }}>
          <div style={{ fontSize: "64px", marginBottom: "16px" }}>📄</div>
          <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "8px" }}>
            PDF Report
          </h2>
          <p style={{ color: "#64748b", fontSize: "14px", marginBottom: "24px" }}>
            Download all voters as formatted PDF document
          </p>
          <button
            onClick={downloadPDF}
            style={{
              padding: "12px 32px", background: "#dc2626",
              color: "#fff", border: "none", borderRadius: "8px",
              fontSize: "15px", fontWeight: "600", cursor: "pointer"
            }}
          >
            ⬇️ Download PDF
          </button>
        </div>
      </div>
    </div>
  );
}