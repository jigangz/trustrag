import React from "react";
import { Search, Cpu, ShieldCheck } from "lucide-react";

const STAGES = {
  retrieving: {
    label: "Searching relevant documents...",
    Icon: Search,
    color: "text-blue-400",
  },
  generating: {
    label: "Generating answer...",
    Icon: Cpu,
    color: "text-amber-400",
  },
  verifying_trust: {
    label: "Verifying trust score...",
    Icon: ShieldCheck,
    color: "text-emerald-400",
  },
};

export default function StatusIndicator({ stage }) {
  const config = STAGES[stage];
  if (!config) return null;

  const { label, Icon, color } = config;

  return (
    <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-900/60 border border-slate-800 animate-fade-in">
      <Icon className={`w-4 h-4 ${color} animate-pulse`} />
      <span className="text-xs text-slate-400">{label}</span>
    </div>
  );
}
