import { useState, useCallback } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export function useQuery() {
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const submitQuery = useCallback(async (question) => {
    try {
      setLoading(true);
      setError(null);
      setResponse(null);
      const res = await axios.post(`${API}/api/query`, { question });
      setResponse(res.data);
      return res.data;
    } catch (err) {
      const msg = err.response?.data?.detail || "Query failed";
      setError(msg);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setResponse(null);
    setError(null);
  }, []);

  return { response, loading, error, submitQuery, clear };
}
