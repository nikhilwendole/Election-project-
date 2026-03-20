import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Voters from "./pages/Voters";
import Vote from "./pages/Vote";
import Reports from "./pages/Reports";
import Layout from "./components/Layout";

function App() {
  const isLoggedIn = () => !!localStorage.getItem("admin");

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/vote" element={<Vote />} />
        <Route
          path="/dashboard"
          element={isLoggedIn() ? <Layout><Dashboard /></Layout> : <Navigate to="/" />}
        />
        <Route
          path="/voters"
          element={isLoggedIn() ? <Layout><Voters /></Layout> : <Navigate to="/" />}
        />
        <Route
          path="/reports"
          element={isLoggedIn() ? <Layout><Reports /></Layout> : <Navigate to="/" />}
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;