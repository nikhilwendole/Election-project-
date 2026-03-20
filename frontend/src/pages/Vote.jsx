// import { useRef, useState, useCallback } from "react";
// import Webcam from "react-webcam";
// import axios from "axios";
// import { useNavigate } from "react-router-dom";

// export default function Vote() {
//   const webcamRef              = useRef(null);
//   const navigate               = useNavigate();
//   const [status,   setStatus]  = useState("idle");
//   const [message,  setMessage] = useState("");
//   const [result,   setResult]  = useState(null);
//   const [name,     setName]    = useState("");
//   const [voterNo,  setVoterNo] = useState("");
//   const [encoding, setEncoding]= useState(null);
//   const [loading,  setLoading] = useState(false);

//   const capture = useCallback(async () => {
//     const image = webcamRef.current?.getScreenshot();
//     if (!image) return;

//     setLoading(true);
//     setStatus("scanning");
//     setMessage("Scanning face...");

//     try {
//       const res = await axios.post("http://localhost:5000/api/verify-face", { image });
//       const data = res.data;

//       if (data.status === "no_face") {
//         setStatus("error");
//         setMessage("❌ No face detected. Please look at camera.");
//       } else if (data.status === "multiple_faces") {
//         setStatus("error");
//         setMessage("❌ Multiple faces detected. Only one voter at a time.");
//       } else if (data.status === "duplicate") {
//         setStatus("duplicate");
//         setResult(data);
//         setMessage(`⚠️ Already voted! ${data.voter.name} (${data.confidence}% match)`);
//       } else if (data.status === "new") {
//         setStatus("new");
//         setEncoding(data.encoding);
//         setMessage("✅ New voter verified! Enter your details below.");
//       }
//     } catch {
//       setStatus("error");
//       setMessage("❌ Server error. Make sure backend is running.");
//     }
//     setLoading(false);
//   }, [webcamRef]);

//   const castVote = async () => {
//     if (!name.trim()) {
//       alert("Please enter your name");
//       return;
//     }
//     setLoading(true);
//     try {
//       const res = await axios.post("http://localhost:5000/api/cast-vote", {
//         name, voter_number: voterNo, encoding
//       });
//       if (res.data.success) {
//         setStatus("voted");
//         setResult(res.data);
//         setMessage("✅ Vote cast successfully!");
//       }
//     } catch {
//       setMessage("❌ Failed to cast vote.");
//     }
//     setLoading(false);
//   };

//   const reset = () => {
//     setStatus("idle");
//     setMessage("");
//     setResult(null);
//     setName("");
//     setVoterNo("");
//     setEncoding(null);
//   };

//   const statusColors = {
//     idle:      "#3b82f6",
//     scanning:  "#f59e0b",
//     new:       "#10b981",
//     duplicate: "#ef4444",
//     voted:     "#10b981",
//     error:     "#ef4444"
//   };

//   return (
//     <div style={{
//       minHeight: "100vh", background: "#0f172a",
//       display: "flex", flexDirection: "column",
//       alignItems: "center", padding: "32px"
//     }}>
//       {/* Header */}
//       <div style={{ textAlign: "center", marginBottom: "32px" }}>
//         <div style={{ fontSize: "40px" }}>🗳️</div>
//         <h1 style={{ fontSize: "28px", fontWeight: "700", color: "#fff", marginTop: "8px" }}>
//           Voting Booth
//         </h1>
//         <p style={{ color: "#64748b" }}>Face verification required to cast vote</p>
//       </div>

//       <div style={{ display: "flex", gap: "32px", width: "100%", maxWidth: "900px" }}>
//         {/* Camera */}
//         <div style={{ flex: 1 }}>
//           <div style={{
//             background: "#1e293b", borderRadius: "16px",
//             padding: "20px", border: "1px solid #334155"
//           }}>
//             <Webcam
//               ref={webcamRef}
//               screenshotFormat="image/jpeg"
//               style={{ width: "100%", borderRadius: "12px" }}
//               mirrored
//             />

//             {/* Status bar */}
//             {message && (
//               <div style={{
//                 marginTop: "16px", padding: "12px 16px",
//                 background: "#0f172a", borderRadius: "8px",
//                 borderLeft: `4px solid ${statusColors[status] || "#3b82f6"}`,
//                 color: "#e2e8f0", fontSize: "14px"
//               }}>
//                 {message}
//               </div>
//             )}

//             {/* Capture button */}
//             {(status === "idle" || status === "error") && (
//               <button
//                 onClick={capture}
//                 disabled={loading}
//                 style={{
//                   width: "100%", marginTop: "16px", padding: "14px",
//                   background: "#3b82f6", color: "#fff", border: "none",
//                   borderRadius: "10px", fontSize: "16px", fontWeight: "600",
//                   cursor: "pointer"
//                 }}
//               >
//                 {loading ? "Scanning..." : "📸 Capture & Verify Face"}
//               </button>
//             )}

//             {/* Reset button */}
//             {status !== "idle" && status !== "voted" && (
//               <button
//                 onClick={reset}
//                 style={{
//                   width: "100%", marginTop: "10px", padding: "12px",
//                   background: "transparent", color: "#94a3b8",
//                   border: "1px solid #334155", borderRadius: "10px",
//                   fontSize: "14px", cursor: "pointer"
//                 }}
//               >
//                 🔄 Try Again
//               </button>
//             )}
//           </div>
//         </div>

//         {/* Right panel */}
//         <div style={{ width: "300px" }}>

//           {/* NEW VOTER — enter details */}
//           {status === "new" && (
//             <div style={{
//               background: "#1e293b", borderRadius: "16px",
//               padding: "24px", border: "1px solid #10b981"
//             }}>
//               <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#10b981", marginBottom: "20px" }}>
//                 ✅ New Voter
//               </h2>
//               <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
//                 <div>
//                   <label style={{ fontSize: "13px", color: "#94a3b8", display: "block", marginBottom: "6px" }}>
//                     Full Name *
//                   </label>
//                   <input
//                     value={name}
//                     onChange={e => setName(e.target.value)}
//                     placeholder="Enter your name"
//                     style={{
//                       width: "100%", padding: "10px 14px",
//                       background: "#0f172a", border: "1px solid #334155",
//                       borderRadius: "8px", color: "#fff", fontSize: "14px", outline: "none"
//                     }}
//                   />
//                 </div>
//                 <div>
//                   <label style={{ fontSize: "13px", color: "#94a3b8", display: "block", marginBottom: "6px" }}>
//                     Voter Number
//                   </label>
//                   <input
//                     value={voterNo}
//                     onChange={e => setVoterNo(e.target.value)}
//                     placeholder="e.g. VN-12345"
//                     style={{
//                       width: "100%", padding: "10px 14px",
//                       background: "#0f172a", border: "1px solid #334155",
//                       borderRadius: "8px", color: "#fff", fontSize: "14px", outline: "none"
//                     }}
//                   />
//                 </div>
//                 <button
//                   onClick={castVote}
//                   disabled={loading}
//                   style={{
//                     padding: "13px", background: "#10b981",
//                     color: "#fff", border: "none", borderRadius: "8px",
//                     fontSize: "15px", fontWeight: "600",
//                     cursor: "pointer", marginTop: "8px"
//                   }}
//                 >
//                   {loading ? "Casting Vote..." : "🗳️ Cast Vote"}
//                 </button>
//               </div>
//             </div>
//           )}

//           {/* DUPLICATE */}
//           {status === "duplicate" && result && (
//             <div style={{
//               background: "#1e293b", borderRadius: "16px",
//               padding: "24px", border: "1px solid #ef4444"
//             }}>
//               <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#ef4444", marginBottom: "16px" }}>
//                 🚫 Already Voted!
//               </h2>
//               <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
//                 {[
//                   ["Name",       result.voter.name],
//                   ["Voter ID",   result.voter.voter_id],
//                   ["Voted At",   result.voter.voted_at?.slice(0, 16)],
//                   ["Confidence", `${result.confidence}%`],
//                 ].map(([k, v]) => (
//                   <div key={k} style={{
//                     background: "#0f172a", borderRadius: "8px", padding: "10px 14px"
//                   }}>
//                     <div style={{ fontSize: "11px", color: "#64748b" }}>{k}</div>
//                     <div style={{ fontSize: "14px", color: "#e2e8f0", marginTop: "2px" }}>{v}</div>
//                   </div>
//                 ))}
//               </div>
//             </div>
//           )}

//           {/* VOTED SUCCESS */}
//           {status === "voted" && result && (
//             <div style={{
//               background: "#1e293b", borderRadius: "16px",
//               padding: "24px", border: "1px solid #10b981",
//               textAlign: "center"
//             }}>
//               <div style={{ fontSize: "48px", marginBottom: "16px" }}>🎉</div>
//               <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#10b981", marginBottom: "16px" }}>
//                 Vote Cast!
//               </h2>
//               {[
//                 ["Name",       result.voter_name],
//                 ["Voter ID",   result.voter_id],
//                 ["Voter No",   result.voter_number],
//               ].map(([k, v]) => (
//                 <div key={k} style={{
//                   background: "#0f172a", borderRadius: "8px",
//                   padding: "10px 14px", marginBottom: "8px", textAlign: "left"
//                 }}>
//                   <div style={{ fontSize: "11px", color: "#64748b" }}>{k}</div>
//                   <div style={{ fontSize: "14px", color: "#e2e8f0", marginTop: "2px" }}>{v}</div>
//                 </div>
//               ))}
//               <button
//                 onClick={reset}
//                 style={{
//                   width: "100%", marginTop: "16px", padding: "12px",
//                   background: "#3b82f6", color: "#fff", border: "none",
//                   borderRadius: "8px", cursor: "pointer", fontSize: "14px"
//                 }}
//               >
//                 Next Voter →
//               </button>
//             </div>
//           )}

//           {/* Admin link */}
//           <button
//             onClick={() => navigate("/")}
//             style={{
//               width: "100%", marginTop: "16px", padding: "12px",
//               background: "transparent", color: "#64748b",
//               border: "1px solid #334155", borderRadius: "10px",
//               fontSize: "13px", cursor: "pointer"
//             }}
//           >
//             🔐 Admin Login
//           </button>
//         </div>
//       </div>
//     </div>
//   );
// }


// import { useRef, useState, useCallback } from "react";
// import Webcam from "react-webcam";
// import axios from "axios";
// import { useNavigate } from "react-router-dom";

// export default function Vote() {
//   const webcamRef = useRef(null);
//   const navigate = useNavigate();
//   const [status, setStatus] = useState("idle"); // idle, scanning, new, duplicate, voted, error
//   const [message, setMessage] = useState("");
//   const [result, setResult] = useState(null);
//   const [name, setName] = useState("");
//   const [voterNo, setVoterNo] = useState("");
//   const [encoding, setEncoding] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const capture = useCallback(async () => {
//     const image = webcamRef.current?.getScreenshot();
//     if (!image) return;

//     setLoading(true);
//     setStatus("scanning");
//     setMessage("Analyzing biometric features...");

//     try {
//       const res = await axios.post("http://localhost:5000/api/verify-face", { image });
//       const data = res.data;

//       if (data.status === "no_face") {
//         setStatus("error");
//         setMessage("No face detected. Please center yourself in the frame.");
//       } else if (data.status === "multiple_faces") {
//         setStatus("error");
//         setMessage("Multiple faces detected. Booth must be private.");
//       } else if (data.status === "duplicate") {
//         setStatus("duplicate");
//         setResult(data);
//         setMessage(`Identity verified: Record already exists.`);
//       } else if (data.status === "new") {
//         setStatus("new");
//         setEncoding(data.encoding);
//         setMessage("Identity verified. Please complete registration.");
//       }
//     } catch {
//       setStatus("error");
//       setMessage("System offline. Please contact an administrator.");
//     }
//     setLoading(false);
//   }, [webcamRef]);

//   const castVote = async () => {
//     if (!name.trim()) return alert("Name is required");
//     setLoading(true);
//     try {
//       const res = await axios.post("http://localhost:5000/api/cast-vote", {
//         name, voter_number: voterNo, encoding
//       });
//       if (res.data.success) {
//         setStatus("voted");
//         setResult(res.data);
//       }
//     } catch {
//       setStatus("error");
//       setMessage("Failed to secure vote. Try again.");
//     }
//     setLoading(false);
//   };

//   const reset = () => {
//     setStatus("idle");
//     setMessage("");
//     setResult(null);
//     setName("");
//     setVoterNo("");
//     setEncoding(null);
//   };

//   return (
//     <div className="min-h-screen bg-[#020617] text-slate-200 font-sans p-4 md:p-8 flex flex-col items-center">
      
//       {/* Header */}
//       <div className="text-center mb-10">
//         <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-600/20 text-blue-500 text-3xl mb-4 shadow-lg shadow-blue-500/10 border border-blue-500/20">
//           🗳️
//         </div>
//         <h1 className="text-3xl font-black tracking-tight text-white md:text-4xl">
//           Secure Voting Portal
//         </h1>
//         <p className="text-slate-400 mt-2 font-medium">Biometric Authentication Required</p>
//       </div>

//       <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
//         {/* Left Column: Camera View */}
//         <div className="lg:col-span-7">
//           <div className="relative bg-slate-900 rounded-3xl p-3 border border-slate-800 shadow-2xl overflow-hidden">
//             {/* Scanning Overlay Animation */}
//             {status === "scanning" && (
//               <div className="absolute inset-0 z-10 pointer-events-none">
//                 <div className="w-full h-1 bg-blue-500 shadow-[0_0_15px_#3b82f6] animate-scan shadow-blue-500/50"></div>
//                 <div className="absolute inset-0 bg-blue-500/5 animate-pulse"></div>
//               </div>
//             )}
            
//             <div className="relative rounded-2xl overflow-hidden aspect-video bg-black">
//               <Webcam
//                 ref={webcamRef}
//                 screenshotFormat="image/jpeg"
//                 className={`w-full h-full object-cover transition-opacity duration-500 ${status === 'voted' ? 'opacity-20' : 'opacity-100'}`}
//                 mirrored
//               />
              
//               {/* Corner Accents */}
//               <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-blue-500/50 rounded-tl-lg"></div>
//               <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-blue-500/50 rounded-tr-lg"></div>
//               <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-blue-500/50 rounded-bl-lg"></div>
//               <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-blue-500/50 rounded-br-lg"></div>
//             </div>

//             {/* Status Feedback */}
//             {message && (
//               <div className={`mt-4 p-4 rounded-xl border flex items-center gap-3 transition-all duration-300 ${
//                 status === 'error' || status === 'duplicate' ? 'bg-red-500/10 border-red-500/20 text-red-400' : 
//                 status === 'new' ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' :
//                 'bg-blue-500/10 border-blue-500/20 text-blue-400'
//               }`}>
//                 <span className="text-xl">
//                   {status === 'error' ? '⚠️' : status === 'duplicate' ? '🚫' : status === 'new' ? '✅' : '📡'}
//                 </span>
//                 <p className="text-sm font-semibold tracking-wide uppercase italic">{message}</p>
//               </div>
//             )}

//             {/* Action Buttons */}
//             {(status === "idle" || status === "error") && (
//               <button
//                 onClick={capture}
//                 disabled={loading}
//                 className="w-full mt-4 py-4 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white rounded-xl font-bold text-lg transition-all shadow-lg shadow-blue-600/20 active:scale-[0.98]"
//               >
//                 {loading ? "Initializing..." : "Verify Identity"}
//               </button>
//             )}

//             {(status !== "idle" && status !== "voted") && (
//               <button onClick={reset} className="w-full mt-3 py-3 text-slate-400 hover:text-white transition-colors text-sm font-medium">
//                 Try Different Angle
//               </button>
//             )}
//           </div>
//         </div>

//         {/* Right Column: Information/Forms */}
//         <div className="lg:col-span-5 space-y-6">
          
//           {/* Default Waiting State */}
//           {status === "idle" && (
//             <div className="bg-slate-900/50 border border-slate-800 p-8 rounded-3xl text-center flex flex-col items-center justify-center min-h-[300px]">
//               <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center text-2xl mb-4 animate-bounce">
//                 👤
//               </div>
//               <h3 className="text-xl font-bold text-white mb-2">Awaiting Verification</h3>
//               <p className="text-slate-500 text-sm leading-relaxed">
//                 Please look directly at the camera. Ensure your face is well-lit and not covered by masks or sunglasses.
//               </p>
//             </div>
//           )}

//           {/* New Voter Form */}
//           {status === "new" && (
//             <div className="bg-slate-900 border border-emerald-500/30 p-8 rounded-3xl shadow-2xl animate-in fade-in slide-in-from-right-4">
//               <h2 className="text-2xl font-bold text-emerald-400 mb-6">Eligible to Vote</h2>
//               <div className="space-y-4">
//                 <div className="space-y-1.5">
//                   <label className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">Full Legal Name</label>
//                   <input
//                     value={name}
//                     onChange={e => setName(e.target.value)}
//                     className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 focus:outline-none focus:border-emerald-500/50 transition-colors"
//                     placeholder="Enter full name"
//                   />
//                 </div>
//                 <div className="space-y-1.5">
//                   <label className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">Voter ID (Optional)</label>
//                   <input
//                     value={voterNo}
//                     onChange={e => setVoterNo(e.target.value)}
//                     className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 focus:outline-none focus:border-emerald-500/50 transition-colors"
//                     placeholder="V-00000"
//                   />
//                 </div>
//                 <button
//                   onClick={castVote}
//                   disabled={loading}
//                   className="w-full py-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-bold text-lg mt-4 shadow-lg shadow-emerald-600/20 transition-all active:scale-[0.98]"
//                 >
//                   Confirm & Submit Vote
//                 </button>
//               </div>
//             </div>
//           )}

//        {/* Duplicate Detected */}
// {status === "duplicate" && result && (
//   <div className="bg-slate-900 border border-red-500/30 p-8 rounded-3xl shadow-2xl">
//     <div className="w-12 h-12 bg-red-500/20 text-red-500 rounded-full flex items-center justify-center text-xl mb-4">
//       🚫
//     </div>
//     <h2 className="text-2xl font-bold text-red-500 mb-2">
//       Integrity Alert
//     </h2>
//     <p className="text-slate-400 text-sm mb-6">
//       Our records indicate this identity has already participated in this election.
//     </p>

//     <div className="space-y-2">
//       {[
//         ["Voter",    result.voter.name],
//         ["Ref Code", result.voter.voter_id],
//         ["Logged At",result.voter.voted_at?.slice(11, 19)],
//         ["Accuracy", `${result.confidence}%`]
//       ].map(([k, v]) => (
//         <div key={k} className="flex justify-between p-3 bg-slate-950 rounded-lg border border-slate-800/50 text-sm">
//           <span className="text-slate-500 uppercase text-[10px] font-bold tracking-tighter self-center">
//             {k}
//           </span>
//           <span className="text-slate-200 font-mono">{v}</span>
//         </div>
//       ))}
//     </div>

//     <button
//       onClick={reset}
//       className="w-full mt-6 py-3 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 rounded-xl text-sm font-semibold cursor-pointer transition-all"
//     >
//       🔄 Try Different Person
//     </button>
//   </div>
// )}
//           {/* Success State */}
//           {status === "voted" && result && (
//             <div className="bg-slate-900 border border-blue-500/30 p-8 rounded-3xl shadow-2xl text-center">
//               <div className="text-6xl mb-4">🏆</div>
//               <h2 className="text-2xl font-black text-white mb-2">Vote Recorded</h2>
//               <p className="text-blue-400 font-mono text-sm mb-6 tracking-widest uppercase">Encryption Successful</p>
//               <button
//                 onClick={reset}
//                 className="w-full py-4 bg-slate-100 hover:bg-white text-slate-950 rounded-xl font-bold transition-all"
//               >
//                 Clear Station for Next Voter
//               </button>
//             </div>
//           )}

//           <button
//             onClick={() => navigate("/")}
//             className="w-full py-3 bg-transparent border border-slate-800 text-slate-500 hover:text-slate-300 hover:bg-slate-900 rounded-xl text-xs font-bold uppercase tracking-widest transition-all"
//           >
//             🔒 Admin Access Only
//           </button>
//         </div>
//       </div>

//       <style jsx>{`
//         @keyframes scan {
//           0% { top: 0; opacity: 0; }
//           10% { opacity: 1; }
//           90% { opacity: 1; }
//           100% { top: 100%; opacity: 0; }
//         }
//         .animate-scan {
//           position: absolute;
//           width: 100%;
//           animation: scan 2s linear infinite;
//         }
//       `}</style>
//     </div>
//   );
// }






import { useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import { useNavigate } from "react-router-dom";

export default function Vote() {
  const webcamRef = useRef(null);
  const navigate = useNavigate();
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [name, setName] = useState("");
  const [voterNo, setVoterNo] = useState("");
  const [encoding, setEncoding] = useState(null);
  const [loading, setLoading] = useState(false);

  // ── 🔊 ALERT SOUND (duplicate) ──────────────────────────
  const playAlertSound = () => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const beep = (freq, start, duration) => {
        const osc  = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = "square";
        osc.frequency.setValueAtTime(freq, ctx.currentTime + start);
        gain.gain.setValueAtTime(0.25, ctx.currentTime + start);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + duration);
        osc.start(ctx.currentTime + start);
        osc.stop(ctx.currentTime + start + duration);
      };
      beep(880, 0.0, 0.25);   // High
      beep(660, 0.3, 0.25);   // Mid
      beep(440, 0.6, 0.35);   // Low
      beep(880, 0.0, 0.25);
      beep(660, 0.3, 0.25);
      beep(440, 0.6, 0.35);
    } catch (e) {
      console.warn("Audio error:", e);
    }
  };

  // ── 🔊 SUCCESS SOUND (new vote) ─────────────────────────
  const playSuccessSound = () => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const beep = (freq, start, duration) => {
        const osc  = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, ctx.currentTime + start);
        gain.gain.setValueAtTime(0.2, ctx.currentTime + start);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + duration);
        osc.start(ctx.currentTime + start);
        osc.stop(ctx.currentTime + start + duration);
      };
      beep(523, 0.0,  0.15);  // C
      beep(659, 0.18, 0.15);  // E
      beep(784, 0.36, 0.15);  // G
      beep(1047,0.54, 0.3);   // High C
    } catch (e) {
      console.warn("Audio error:", e);
    }
  };

  // ── 🔊 SCAN SOUND (scanning) ────────────────────────────
  const playScanSound = () => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = "sine";
      osc.frequency.setValueAtTime(300, ctx.currentTime);
      osc.frequency.linearRampToValueAtTime(800, ctx.currentTime + 0.4);
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.4);
    } catch (e) {
      console.warn("Audio error:", e);
    }
  };

  // ── CAPTURE ──────────────────────────────────────────────
  const capture = useCallback(async () => {
    const image = webcamRef.current?.getScreenshot();
    if (!image) return;

    setLoading(true);
    setStatus("scanning");
    setMessage("Analyzing biometric features...");
    playScanSound();

    try {
      const res  = await axios.post("http://localhost:5000/api/verify-face", { image });
      const data = res.data;

      if (data.status === "no_face") {
        setStatus("error");
        setMessage("No face detected. Please center yourself in the frame.");
      } else if (data.status === "multiple_faces") {
        setStatus("error");
        setMessage("Multiple faces detected. Booth must be private.");
      } else if (data.status === "duplicate") {
        setStatus("duplicate");
        setResult(data);
        setMessage("Identity verified: Record already exists.");
        playAlertSound();   // 🚨 ALARM
      } else if (data.status === "new") {
        setStatus("new");
        setEncoding(data.encoding);
        setMessage("Identity verified. Please complete registration.");
      }
    } catch {
      setStatus("error");
      setMessage("System offline. Please contact an administrator.");
    }
    setLoading(false);
  }, [webcamRef]);

  // ── CAST VOTE ────────────────────────────────────────────
  const castVote = async () => {
    if (!name.trim()) return alert("Name is required");
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/api/cast-vote", {
        name, voter_number: voterNo, encoding
      });
      if (res.data.success) {
        setStatus("voted");
        setResult(res.data);
        playSuccessSound();  // 🎉 SUCCESS
      }
    } catch {
      setStatus("error");
      setMessage("Failed to secure vote. Try again.");
    }
    setLoading(false);
  };

  const reset = () => {
    setStatus("idle");
    setMessage("");
    setResult(null);
    setName("");
    setVoterNo("");
    setEncoding(null);
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-200 font-sans p-4 md:p-8 flex flex-col items-center">

      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-600/20 text-blue-500 text-3xl mb-4 shadow-lg shadow-blue-500/10 border border-blue-500/20">
          🗳️
        </div>
        <h1 className="text-3xl font-black tracking-tight text-white md:text-4xl">
          Secure Voting Portal
        </h1>
        <p className="text-slate-400 mt-2 font-medium">Biometric Authentication Required</p>
      </div>

      <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

        {/* Left — Camera */}
        <div className="lg:col-span-7">
          <div className="relative bg-slate-900 rounded-3xl p-3 border border-slate-800 shadow-2xl overflow-hidden">

            {/* Scanning overlay */}
            {status === "scanning" && (
              <div className="absolute inset-0 z-10 pointer-events-none">
                <div className="w-full h-1 bg-blue-500 shadow-[0_0_15px_#3b82f6] animate-scan"></div>
                <div className="absolute inset-0 bg-blue-500/5 animate-pulse"></div>
              </div>
            )}

            {/* Duplicate flashing overlay */}
            {status === "duplicate" && (
              <div className="absolute inset-0 z-10 pointer-events-none rounded-3xl border-4 border-red-500 animate-pulse"></div>
            )}

            <div className="relative rounded-2xl overflow-hidden aspect-video bg-black">
              <Webcam
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className={`w-full h-full object-cover transition-opacity duration-500 ${
                  status === "voted" ? "opacity-20" :
                  status === "duplicate" ? "opacity-60" : "opacity-100"
                }`}
                mirrored
              />

              {/* Corner accents */}
              <div className={`absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 rounded-tl-lg transition-colors ${status === "duplicate" ? "border-red-500" : "border-blue-500/50"}`}></div>
              <div className={`absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 rounded-tr-lg transition-colors ${status === "duplicate" ? "border-red-500" : "border-blue-500/50"}`}></div>
              <div className={`absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 rounded-bl-lg transition-colors ${status === "duplicate" ? "border-red-500" : "border-blue-500/50"}`}></div>
              <div className={`absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 rounded-br-lg transition-colors ${status === "duplicate" ? "border-red-500" : "border-blue-500/50"}`}></div>

              {/* Duplicate big warning on camera */}
              {status === "duplicate" && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-red-950/60 backdrop-blur-sm">
                  <div className="text-6xl mb-3 animate-bounce">🚫</div>
                  <div className="text-red-400 font-black text-2xl tracking-widest uppercase">
                    DUPLICATE
                  </div>
                  <div className="text-red-300 text-sm mt-1 font-mono">
                    Vote Already Recorded
                  </div>
                </div>
              )}

              {/* Success overlay on camera */}
              {status === "voted" && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-emerald-950/60 backdrop-blur-sm">
                  <div className="text-6xl mb-3">✅</div>
                  <div className="text-emerald-400 font-black text-2xl tracking-widest uppercase">
                    Vote Cast!
                  </div>
                </div>
              )}
            </div>

            {/* Status message */}
            {message && (
              <div className={`mt-4 p-4 rounded-xl border flex items-center gap-3 transition-all duration-300 ${
                status === "error"     ? "bg-red-500/10 border-red-500/20 text-red-400" :
                status === "duplicate" ? "bg-red-500/10 border-red-500/20 text-red-400" :
                status === "new"       ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" :
                                         "bg-blue-500/10 border-blue-500/20 text-blue-400"
              }`}>
                <span className="text-xl">
                  {status === "error"     ? "⚠️" :
                   status === "duplicate" ? "🚫" :
                   status === "new"       ? "✅" : "📡"}
                </span>
                <p className="text-sm font-semibold tracking-wide uppercase italic">{message}</p>
              </div>
            )}

            {/* Capture button */}
            {(status === "idle" || status === "error") && (
              <button
                onClick={capture}
                disabled={loading}
                className="w-full mt-4 py-4 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white rounded-xl font-bold text-lg transition-all shadow-lg shadow-blue-600/20 active:scale-[0.98]"
              >
                {loading ? "Initializing..." : "Verify Identity"}
              </button>
            )}

            {(status !== "idle" && status !== "voted") && (
              <button
                onClick={reset}
                className="w-full mt-3 py-3 text-slate-400 hover:text-white transition-colors text-sm font-medium"
              >
                Try Different Angle
              </button>
            )}
          </div>
        </div>

        {/* Right — Info Panel */}
        <div className="lg:col-span-5 space-y-6">

          {/* Idle */}
          {status === "idle" && (
            <div className="bg-slate-900/50 border border-slate-800 p-8 rounded-3xl text-center flex flex-col items-center justify-center min-h-[300px]">
              <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center text-2xl mb-4 animate-bounce">
                👤
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Awaiting Verification</h3>
              <p className="text-slate-500 text-sm leading-relaxed">
                Please look directly at the camera. Ensure your face is well-lit and not covered.
              </p>
            </div>
          )}

          {/* New Voter */}
          {status === "new" && (
            <div className="bg-slate-900 border border-emerald-500/30 p-8 rounded-3xl shadow-2xl">
              <h2 className="text-2xl font-bold text-emerald-400 mb-6">Eligible to Vote</h2>
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">
                    Full Legal Name
                  </label>
                  <input
                    value={name}
                    onChange={e => setName(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 focus:outline-none focus:border-emerald-500/50 transition-colors text-white"
                    placeholder="Enter full name"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">
                    Voter ID (Optional)
                  </label>
                  <input
                    value={voterNo}
                    onChange={e => setVoterNo(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 focus:outline-none focus:border-emerald-500/50 transition-colors text-white"
                    placeholder="V-00000"
                  />
                </div>
                <button
                  onClick={castVote}
                  disabled={loading}
                  className="w-full py-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-bold text-lg mt-4 shadow-lg shadow-emerald-600/20 transition-all active:scale-[0.98]"
                >
                  {loading ? "Recording..." : "Confirm & Submit Vote"}
                </button>
              </div>
            </div>
          )}

          {/* Duplicate */}
          {status === "duplicate" && result && (
            <div className="bg-slate-900 border border-red-500/30 p-8 rounded-3xl shadow-2xl">
              <div className="w-12 h-12 bg-red-500/20 text-red-500 rounded-full flex items-center justify-center text-xl mb-4 animate-pulse">
                🚫
              </div>
              <h2 className="text-2xl font-bold text-red-500 mb-2">Integrity Alert</h2>
              <p className="text-slate-400 text-sm mb-6">
                Our records indicate this identity has already participated in this election.
              </p>

              <div className="space-y-2">
                {[
                  ["Voter",     result.voter.name],
                  ["Ref Code",  result.voter.voter_id],
                  ["Logged At", result.voter.voted_at?.slice(11, 19)],
                  ["Accuracy",  `${result.confidence}%`],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between p-3 bg-slate-950 rounded-lg border border-slate-800/50 text-sm">
                    <span className="text-slate-500 uppercase text-[10px] font-bold tracking-tighter self-center">
                      {k}
                    </span>
                    <span className="text-slate-200 font-mono">{v}</span>
                  </div>
                ))}
              </div>

              <button
                onClick={reset}
                className="w-full mt-6 py-3 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 rounded-xl text-sm font-semibold cursor-pointer transition-all"
              >
                🔄 Try Different Person
              </button>
            </div>
          )}

          {/* Success */}
          {status === "voted" && result && (
            <div className="bg-slate-900 border border-blue-500/30 p-8 rounded-3xl shadow-2xl text-center">
              <div className="text-6xl mb-4 animate-bounce">🏆</div>
              <h2 className="text-2xl font-black text-white mb-2">Vote Recorded</h2>
              <p className="text-blue-400 font-mono text-sm mb-6 tracking-widest uppercase">
                Encryption Successful
              </p>
              <div className="space-y-2 mb-6 text-left">
                {[
                  ["Name",      result.voter_name],
                  ["Voter ID",  result.voter_id],
                  ["Voter No",  result.voter_number],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between p-3 bg-slate-950 rounded-lg border border-slate-800/50 text-sm">
                    <span className="text-slate-500 uppercase text-[10px] font-bold tracking-tighter self-center">{k}</span>
                    <span className="text-slate-200 font-mono">{v}</span>
                  </div>
                ))}
              </div>
              <button
                onClick={reset}
                className="w-full py-4 bg-slate-100 hover:bg-white text-slate-950 rounded-xl font-bold transition-all"
              >
                Clear Station for Next Voter
              </button>
            </div>
          )}

          <button
            onClick={() => navigate("/")}
            className="w-full py-3 bg-transparent border border-slate-800 text-slate-500 hover:text-slate-300 hover:bg-slate-900 rounded-xl text-xs font-bold uppercase tracking-widest transition-all"
          >
            🔒 Admin Access Only
          </button>
        </div>
      </div>

      <style>{`
        @keyframes scan {
          0%   { top: 0;    opacity: 0; }
          10%  { opacity: 1; }
          90%  { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        .animate-scan {
          position: absolute;
          width: 100%;
          animation: scan 2s linear infinite;
        }
      `}</style>
    </div>
  );
}



{/* Spoof Detected */}
{status === "spoof" && (
  <div className="bg-slate-900 border border-orange-500/30 p-8 rounded-3xl shadow-2xl">
    <div className="w-12 h-12 bg-orange-500/20 text-orange-500 rounded-full flex items-center justify-center text-2xl mb-4 animate-pulse">
      ⚠️
    </div>
    <h2 className="text-2xl font-bold text-orange-500 mb-2">
      Spoof Attempt Detected
    </h2>
    <p className="text-slate-400 text-sm mb-4">
      Our system has detected a non-live face. Please do not use:
    </p>
    <div className="space-y-2 mb-6">
      {["📷 Printed photos", "📱 Phone/screen images", "🎭 Masks or disguises"].map(item => (
        <div key={item} className="flex items-center gap-3 p-3 bg-slate-950 rounded-lg border border-orange-500/20">
          <span className="text-sm text-slate-300">{item}</span>
        </div>
      ))}
    </div>
    <p className="text-slate-500 text-xs mb-6">
      This attempt has been logged. Please present your real face to continue.
    </p>
    <button
      onClick={reset}
      className="w-full py-3 bg-orange-500/10 hover:bg-orange-500/20 text-orange-400 border border-orange-500/30 rounded-xl text-sm font-semibold cursor-pointer transition-all"
    >
      🔄 Try Again with Real Face
    </button>
  </div>
)}