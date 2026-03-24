import React, { useState } from "react";
import { Shield, FileText, Search, ClipboardList, Menu, X } from "lucide-react";
import DocumentList from "./DocumentList";
import DocumentUpload from "./DocumentUpload";
import QueryPanel from "./QueryPanel";
import AuditTimeline from "./AuditTimeline";

const NAV = [
  { id: "query", label: "Query", icon: Search },
  { id: "audit", label: "Audit Log", icon: ClipboardList },
];

export default function Layout() {
  const [view, setView] = useState("query");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? "w-80" : "w-0 overflow-hidden"
        } flex-shrink-0 border-r border-slate-800 bg-slate-925 transition-all duration-300 flex flex-col`}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-slate-800">
          <div className="p-2 rounded-lg bg-emerald-500/10">
            <Shield className="w-6 h-6 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">TrustRAG</h1>
            <p className="text-xs text-slate-500">Construction AI Q&A</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="px-3 py-3 space-y-1">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                view === id
                  ? "bg-slate-800 text-white"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </nav>

        {/* Document section */}
        <div className="flex-1 flex flex-col min-h-0 border-t border-slate-800 mt-2">
          <div className="flex items-center gap-2 px-5 py-3">
            <FileText className="w-4 h-4 text-slate-500" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Documents
            </span>
          </div>
          <div className="px-3 pb-3">
            <DocumentUpload />
          </div>
          <div className="flex-1 overflow-y-auto px-3 pb-3">
            <DocumentList />
          </div>
        </div>
      </aside>

      {/* Main area */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="flex items-center gap-3 px-5 py-3 border-b border-slate-800 bg-slate-925">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 transition-colors"
          >
            {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
          <h2 className="text-sm font-semibold text-slate-300">
            {view === "query" ? "Ask a Question" : "Audit Log"}
          </h2>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {view === "query" ? <QueryPanel /> : <AuditTimeline />}
        </div>
      </main>
    </div>
  );
}
