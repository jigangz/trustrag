import React, { useState } from "react";
import { Clock, ChevronDown, ChevronUp, Search, ShieldCheck, Loader2 } from "lucide-react";
import { useAudit } from "../hooks/useAudit";

function getScoreDot(score) {
  if (score == null) return "bg-slate-600";
  if (score >= 80) return "bg-emerald-500";
  if (score >= 50) return "bg-amber-500";
  return "bg-red-500";
}

function formatDate(dateStr) {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function Skeleton() {
  return (
    <div className="space-y-4">
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="skeleton h-20 rounded-lg" />
      ))}
    </div>
  );
}

function AuditEntry({ entry }) {
  const [expanded, setExpanded] = useState(false);
  const score = entry.confidence_score;

  return (
    <div className="relative pl-8 animate-fade-in">
      {/* Timeline dot */}
      <div
        className={`absolute left-0 top-4 w-3 h-3 rounded-full ${getScoreDot(score)} ring-4 ring-slate-950`}
      />

      <div className="border border-slate-800 rounded-xl bg-slate-900/50 overflow-hidden">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-start justify-between gap-4 p-4 text-left hover:bg-slate-800/30 transition-colors"
        >
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Search className="w-3.5 h-3.5 text-slate-500" />
              <p className="text-sm font-medium text-slate-200 truncate">
                {entry.question}
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-500">
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {formatDate(entry.timestamp || entry.created_at)}
              </span>
              {score != null && (
                <span className="flex items-center gap-1">
                  <ShieldCheck className="w-3 h-3" />
                  Score: {Math.round(score)}
                </span>
              )}
            </div>
          </div>
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-slate-500 mt-1" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-500 mt-1" />
          )}
        </button>

        {expanded && (
          <div className="px-4 pb-4 pt-0 border-t border-slate-800 mt-0 pt-3 space-y-3 animate-fade-in">
            {entry.answer && (
              <div>
                <p className="text-xs font-semibold text-slate-500 mb-1">Answer</p>
                <p className="text-sm text-slate-300 leading-relaxed">{entry.answer}</p>
              </div>
            )}
            {entry.sources?.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-500 mb-1">
                  Sources ({entry.sources.length})
                </p>
                <div className="space-y-1">
                  {entry.sources.map((s, i) => (
                    <p key={i} className="text-xs text-slate-400">
                      {s.document_name}
                      {s.page_number != null && ` (p. ${s.page_number})`}
                    </p>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function AuditTimeline() {
  const { entries, loading, hasMore, loadMore } = useAudit();

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h2 className="text-lg font-bold text-slate-200 mb-6">Query History</h2>

      {loading && !entries.length ? (
        <Skeleton />
      ) : !entries.length ? (
        <div className="text-center py-16">
          <Clock className="w-10 h-10 text-slate-700 mx-auto mb-3" />
          <p className="text-sm text-slate-500">No queries yet</p>
        </div>
      ) : (
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-[5px] top-4 bottom-4 w-px bg-slate-800" />

          <div className="space-y-4">
            {entries.map((entry, i) => (
              <AuditEntry key={entry.id || i} entry={entry} />
            ))}
          </div>

          {hasMore && (
            <div className="mt-6 text-center">
              <button
                onClick={loadMore}
                disabled={loading}
                className="px-4 py-2 text-sm text-slate-400 border border-slate-800 rounded-lg hover:bg-slate-800 hover:text-slate-200 transition-colors disabled:opacity-50"
              >
                {loading ? (
                  <Loader2 className="w-4 h-4 animate-spin inline mr-2" />
                ) : null}
                Load More
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
