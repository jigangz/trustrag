import { useState, useEffect, useCallback } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export function useAudit() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const pageSize = 20;

  const fetchAudit = useCallback(async (pageNum = 1) => {
    try {
      setLoading(true);
      const res = await axios.get(`${API}/api/audit`, {
        params: { page: pageNum, page_size: pageSize },
      });
      const data = res.data.entries || res.data || [];
      if (pageNum === 1) {
        setEntries(data);
      } else {
        setEntries((prev) => [...prev, ...data]);
      }
      setHasMore(data.length === pageSize);
      setPage(pageNum);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to load audit log");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadMore = useCallback(() => {
    if (!loading && hasMore) {
      fetchAudit(page + 1);
    }
  }, [loading, hasMore, page, fetchAudit]);

  useEffect(() => {
    fetchAudit(1);
  }, [fetchAudit]);

  return { entries, loading, error, hasMore, loadMore, refetch: () => fetchAudit(1) };
}
