import { useState, useEffect, useCallback } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export function useDocuments() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);

  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${API}/api/documents`);
      setDocuments(res.data.documents || res.data || []);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to load documents");
    } finally {
      setLoading(false);
    }
  }, []);

  const uploadDocument = useCallback(async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    try {
      setUploading(true);
      setUploadProgress(0);
      setError(null);
      const res = await axios.post(`${API}/api/documents`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          const pct = Math.round((e.loaded * 100) / (e.total || 1));
          setUploadProgress(pct);
        },
      });
      await fetchDocuments();
      return res.data;
    } catch (err) {
      setError(err.response?.data?.detail || "Upload failed");
      throw err;
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }, [fetchDocuments]);

  const deleteDocument = useCallback(async (docId) => {
    try {
      await axios.delete(`${API}/api/documents/${docId}`);
      await fetchDocuments();
    } catch (err) {
      setError(err.response?.data?.detail || "Delete failed");
    }
  }, [fetchDocuments]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  return { documents, loading, uploading, uploadProgress, error, uploadDocument, deleteDocument, refetch: fetchDocuments };
}
