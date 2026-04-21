import React, { useState } from "react";
import { ShieldCheck, ShieldAlert, ShieldX, ChevronDown, ChevronUp } from "lucide-react";

function getConfig(score) {
  if (score >= 80)
    return {
      color: "emerald",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/30",
      bar: "bg-emerald-500",
      text: "text-emerald-400",
      label: "High Confidence",
      Icon: ShieldCheck,
    };
  if (score >= 50)
    return {
      color: "amber",
      bg: "bg-amber-500/10",
      border: "border-amber-500/30",
      bar: "bg-amber-500",
      text: "text-amber-400",
      label: "Medium Confidence",
      Icon: ShieldAlert,
    };
  return {
    color: "red",
    bg: "bg-red-500/10",
    border: "border-red-500/30",
    bar: "bg-red-500",
    text: "text-red-400",
    label: "Low Confidence",
    Icon: ShieldX,
  };
}

export default function ConfidenceBadge({ score = 0, breakdown, verifying = false }) {
  const [expanded, setExpanded] = useState(false);

  if (verifying) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4 animate-fade-in">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-slate-800">
            <ShieldCheck className="w-7 h-7 text-slate-500 animate-pulse" />
          </div>
          <div>
            <span className="text-lg font-semibold text-slate-400">--</span>
            <span className="text-sm text-slate-600 ml-1">/100</span>
            <p className="text-xs font-medium text-slate-500">Verifying trust...</p>
          </div>
        </div>
        <div className="mt-3 h-2 bg-slate-800 rounded-full overflow-hidden">
          <div className="h-full bg-slate-600 rounded-full animate-pulse w-1/3" />
        </div>
      </div>
    );
  }

  const cfg = getConfig(score);
  const { bg, border, bar, text, label, Icon } = cfg;

  return (
    <div className={`rounded-xl border ${border} ${bg} p-4 animate-fade-in`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2.5 rounded-xl ${bg}`}>
            <Icon className={`w-7 h-7 ${text}`} />
          </div>
          <div>
            <div className="flex items-baseline gap-2">
              <span className={`text-3xl font-bold ${text}`}>{Math.round(score)}</span>
              <span className="text-sm text-slate-500">/100</span>
            </div>
            <p className={`text-xs font-medium ${text}`}>{label}</p>
          </div>
        </div>

        {breakdown && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 transition-colors"
          >
            Details
            {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        )}
      </div>

      {/* Progress bar */}
      <div className="mt-3 h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${bar} rounded-full transition-all duration-700 ease-out`}
          style={{ width: `${Math.min(100, Math.max(0, score))}%` }}
        />
      </div>

      {/* Breakdown */}
      {expanded && breakdown && (
        <div className="mt-4 pt-3 border-t border-slate-800 space-y-2.5 animate-fade-in">
          {Object.entries(breakdown).map(([key, value]) => {
            const itemCfg = getConfig(typeof value === "number" ? value : 0);
            return (
              <div key={key}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-slate-400 capitalize">
                    {key.replace(/_/g, " ")}
                  </span>
                  <span className={itemCfg.text}>
                    {typeof value === "number" ? Math.round(value) : value}
                  </span>
                </div>
                {typeof value === "number" && (
                  <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${itemCfg.bar} rounded-full transition-all duration-500`}
                      style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
