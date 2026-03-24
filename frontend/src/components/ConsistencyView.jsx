import React, { useState } from "react";
import { GitCompare, ChevronDown, ChevronUp } from "lucide-react";

function getScoreColor(score) {
  if (score >= 80) return "text-emerald-400";
  if (score >= 50) return "text-amber-400";
  return "text-red-400";
}

function getBarColor(score) {
  if (score >= 80) return "bg-emerald-500";
  if (score >= 50) return "bg-amber-500";
  return "bg-red-500";
}

export default function ConsistencyView({ data }) {
  const [expanded, setExpanded] = useState(false);
  if (!data) return null;

  const { consistency_score, rephrased_queries } = data;
  const score = consistency_score ?? 0;

  return (
    <div className="border border-slate-800 rounded-xl bg-slate-900/50 animate-fade-in">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-slate-800/30 transition-colors rounded-xl"
      >
        <div className="flex items-center gap-3">
          <GitCompare className="w-4 h-4 text-slate-500" />
          <span className="text-sm font-semibold text-slate-300">Consistency Check</span>
          <span className={`text-sm font-bold ${getScoreColor(score)}`}>
            {Math.round(score)}%
          </span>
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-slate-500" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-500" />
        )}
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-4 animate-fade-in">
          {/* Overall bar */}
          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full ${getBarColor(score)} rounded-full transition-all duration-500`}
              style={{ width: `${score}%` }}
            />
          </div>

          {/* Rephrased queries */}
          {rephrased_queries?.map((item, i) => (
            <div key={i} className="border border-slate-800 rounded-lg p-3 space-y-2">
              <p className="text-xs font-medium text-slate-500">
                Rephrased Question {i + 1}
              </p>
              <p className="text-sm text-slate-300 italic">"{item.question}"</p>
              <p className="text-sm text-slate-400 leading-relaxed">{item.answer}</p>
              {item.similarity != null && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">Similarity:</span>
                  <span className={`text-xs font-semibold ${getScoreColor(item.similarity * 100)}`}>
                    {Math.round(item.similarity * 100)}%
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
