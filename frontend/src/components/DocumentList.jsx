import React from "react";
import { FileText, Trash2, Clock } from "lucide-react";
import { useDocuments } from "../hooks/useDocuments";

function Skeleton() {
  return (
    <div className="space-y-2">
      {[1, 2, 3].map((i) => (
        <div key={i} className="skeleton h-16 rounded-lg" />
      ))}
    </div>
  );
}

function formatDate(dateStr) {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function DocumentList() {
  const { documents, loading, deleteDocument } = useDocuments();

  if (loading) return <Skeleton />;

  if (!documents.length) {
    return (
      <div className="text-center py-8">
        <FileText className="w-8 h-8 text-slate-700 mx-auto mb-2" />
        <p className="text-xs text-slate-600">No documents uploaded</p>
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {documents.map((doc) => (
        <div
          key={doc.id || doc.filename}
          className="group flex items-start gap-3 p-3 rounded-lg hover:bg-slate-800/50 transition-colors"
        >
          <div className="p-1.5 rounded bg-slate-800 text-slate-500 mt-0.5">
            <FileText className="w-3.5 h-3.5" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-slate-200 truncate">
              {doc.filename}
            </p>
            <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
              {doc.pages != null && <span>{doc.pages} pages</span>}
              {doc.chunks != null && <span>{doc.chunks} chunks</span>}
              {doc.uploaded_at && (
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {formatDate(doc.uploaded_at)}
                </span>
              )}
            </div>
          </div>
          <button
            onClick={() => deleteDocument(doc.id)}
            className="p-1.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/10 hover:text-red-400 text-slate-600 transition-all"
            title="Delete document"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}
