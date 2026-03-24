import React from "react";
import { FileText, Hash } from "lucide-react";

export default function SourceCard({ source }) {
  const { document_name, page_number, text, similarity_score } = source || {};

  const simPct = similarity_score != null ? Math.round(similarity_score * 100) : null;
  const simColor =
    simPct >= 80 ? "text-emerald-400" : simPct >= 50 ? "text-amber-400" : "text-red-400";

  return (
    <div className="border border-slate-800 rounded-lg p-4 bg-slate-900/50 hover:bg-slate-900 transition-colors animate-fade-in">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="w-4 h-4 text-slate-500 flex-shrink-0" />
          <span className="text-sm font-medium text-slate-200 truncate">
            {document_name || "Unknown Document"}
          </span>
        </div>
        {simPct != null && (
          <span className={`text-xs font-mono font-semibold ${simColor} flex-shrink-0`}>
            {simPct}% match
          </span>
        )}
      </div>

      {page_number != null && (
        <div className="flex items-center gap-1 text-xs text-slate-500 mb-2">
          <Hash className="w-3 h-3" />
          Page {page_number}
        </div>
      )}

      {text && (
        <p className="text-sm text-slate-400 leading-relaxed line-clamp-3 border-l-2 border-slate-700 pl-3">
          {text}
        </p>
      )}
    </div>
  );
}
