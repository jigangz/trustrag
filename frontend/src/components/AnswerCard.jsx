import React from "react";
import { MessageSquare } from "lucide-react";
import ConfidenceBadge from "./ConfidenceBadge";
import SourceCard from "./SourceCard";
import ConsistencyView from "./ConsistencyView";

function AnswerSkeleton() {
  return (
    <div className="space-y-4">
      <div className="skeleton h-6 w-48 rounded" />
      <div className="skeleton h-24 rounded-lg" />
      <div className="skeleton h-16 rounded-lg" />
      <div className="skeleton h-16 rounded-lg" />
    </div>
  );
}

export default function AnswerCard({ response, loading }) {
  if (loading) return <AnswerSkeleton />;
  if (!response) return null;

  const {
    answer,
    confidence_score,
    confidence_breakdown,
    sources,
    consistency_check,
  } = response;

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Confidence */}
      {confidence_score != null && (
        <ConfidenceBadge score={confidence_score} breakdown={confidence_breakdown} />
      )}

      {/* Answer */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <MessageSquare className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-semibold text-slate-300">Answer</h3>
        </div>
        <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
          {answer || "No answer generated."}
        </div>
      </div>

      {/* Sources */}
      {sources?.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-slate-400 mb-3">
            Sources ({sources.length})
          </h3>
          <div className="grid gap-3 md:grid-cols-2">
            {sources.map((src, i) => (
              <SourceCard key={i} source={src} />
            ))}
          </div>
        </div>
      )}

      {/* Consistency */}
      {consistency_check && <ConsistencyView data={consistency_check} />}
    </div>
  );
}
