import React, { useCallback, useState } from "react";
import { Upload, FileUp, CheckCircle, AlertCircle } from "lucide-react";
import { useDocuments } from "../hooks/useDocuments";

export default function DocumentUpload() {
  const { uploadDocument, uploading, uploadProgress, error } = useDocuments();
  const [dragActive, setDragActive] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleFile = useCallback(
    async (file) => {
      if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
        return;
      }
      setSuccess(false);
      try {
        await uploadDocument(file);
        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      } catch {
        // error is handled in hook
      }
    },
    [uploadDocument]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files?.[0];
      handleFile(file);
    },
    [handleFile]
  );

  const onDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const onDragLeave = () => setDragActive(false);

  const onFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  return (
    <div>
      <label
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        className={`flex flex-col items-center gap-2 p-4 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
          dragActive
            ? "border-emerald-400 bg-emerald-400/5"
            : "border-slate-700 hover:border-slate-500 bg-slate-900/50"
        } ${uploading ? "pointer-events-none opacity-60" : ""}`}
      >
        <input
          type="file"
          accept=".pdf"
          onChange={onFileSelect}
          className="hidden"
          disabled={uploading}
        />
        {uploading ? (
          <FileUp className="w-6 h-6 text-emerald-400 animate-pulse" />
        ) : success ? (
          <CheckCircle className="w-6 h-6 text-emerald-400" />
        ) : (
          <Upload className="w-6 h-6 text-slate-500" />
        )}
        <span className="text-xs text-slate-400">
          {uploading
            ? "Uploading..."
            : success
            ? "Upload complete!"
            : "Drop PDF or click to upload"}
        </span>
      </label>

      {/* Progress bar */}
      {uploading && (
        <div className="mt-2 h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-500 rounded-full transition-all duration-300"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
      )}

      {error && (
        <div className="mt-2 flex items-center gap-1.5 text-xs text-red-400">
          <AlertCircle className="w-3 h-3" />
          {error}
        </div>
      )}
    </div>
  );
}
