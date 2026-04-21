import React from "react";
import { XCircle } from "lucide-react";

const ACTIVE_STAGES = ["retrieving", "generating", "verifying_trust"];

export default function CancelButton({ queryStatus, onCancel }) {
  if (!ACTIVE_STAGES.includes(queryStatus)) return null;

  return (
    <button
      onClick={onCancel}
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-red-400 border border-red-500/30 bg-red-500/10 hover:bg-red-500/20 transition-colors"
    >
      <XCircle className="w-3.5 h-3.5" />
      Cancel
    </button>
  );
}
