import React, { useState } from "react";
import { Send, Loader2, HelpCircle } from "lucide-react";
import { useQuery } from "../hooks/useQuery";
import AnswerCard from "./AnswerCard";

const EXAMPLES = [
  "What are the structural load requirements for floor slabs?",
  "What safety measures are required for scaffolding?",
  "What are the concrete curing time specifications?",
];

export default function QueryPanel() {
  const [question, setQuestion] = useState("");
  const { response, loading, error, submitQuery } = useQuery();

  const handleSubmit = (e) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || loading) return;
    submitQuery(q);
  };

  const askExample = (q) => {
    setQuestion(q);
    submitQuery(q);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Query input */}
      <form onSubmit={handleSubmit} className="relative">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your construction documents..."
          disabled={loading}
          className="w-full bg-slate-900 border border-slate-700 rounded-xl py-4 pl-5 pr-14 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/25 transition-all disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white transition-colors"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </form>

      {/* Example questions */}
      {!response && !loading && (
        <div className="animate-fade-in">
          <div className="flex items-center gap-2 mb-3">
            <HelpCircle className="w-4 h-4 text-slate-600" />
            <span className="text-xs text-slate-600">Example questions</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((q) => (
              <button
                key={q}
                onClick={() => askExample(q)}
                className="text-xs px-3 py-2 rounded-lg border border-slate-800 text-slate-400 hover:border-slate-600 hover:text-slate-300 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-sm text-red-400 animate-fade-in">
          {error}
        </div>
      )}

      {/* Answer */}
      <AnswerCard response={response} loading={loading} />
    </div>
  );
}
