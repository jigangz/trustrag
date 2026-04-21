import React from "react";
import { MessageSquare } from "lucide-react";

const ACTIVE_STAGES = ["retrieving", "generating", "verifying_trust"];

export default function StreamingAnswer({ answer, queryStatus }) {
  const showCursor = ACTIVE_STAGES.includes(queryStatus);

  if (!answer && !showCursor) return null;

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 animate-fade-in">
      <div className="flex items-center gap-2 mb-3">
        <MessageSquare className="w-4 h-4 text-emerald-400" />
        <h3 className="text-sm font-semibold text-slate-300">Answer</h3>
      </div>
      <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
        {answer || ""}
        {showCursor && (
          <span className="inline-block w-2 h-4 ml-0.5 bg-emerald-400 animate-pulse rounded-sm" />
        )}
      </div>
    </div>
  );
}
