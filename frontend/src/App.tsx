import { NavLink, Route, Routes } from "react-router-dom";

import Analysis from "./pages/Analysis";
import Dashboard from "./pages/Dashboard";
import Documents from "./pages/Documents";
import Embeddings from "./pages/Embeddings";
import Extraction from "./pages/Extraction";
import Reconciliation from "./pages/Reconciliation";
import Search from "./pages/Search";
import SystemHealth from "./pages/SystemHealth";

export default function App() {
  return (
    <div className="layout">
      <header className="topbar">
        <span className="brand">Helix</span>
        <nav>
          <NavLink to="/" end>
            Dashboard
          </NavLink>
          <NavLink to="/analysis">Analysis</NavLink>
          <NavLink to="/documents">Documents</NavLink>
          <NavLink to="/embeddings">Embeddings</NavLink>
          <NavLink to="/search">Search</NavLink>
          <NavLink to="/extraction">Extraction</NavLink>
          <NavLink to="/reconciliation">Reconciliation</NavLink>
          <NavLink to="/health">System Health</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/embeddings" element={<Embeddings />} />
          <Route path="/search" element={<Search />} />
          <Route path="/extraction" element={<Extraction />} />
          <Route path="/reconciliation" element={<Reconciliation />} />
          <Route path="/health" element={<SystemHealth />} />
        </Routes>
      </main>
    </div>
  );
}
