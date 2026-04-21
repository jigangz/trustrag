import { useState, useEffect, useRef, useCallback } from "react";
import { TrustRAGWebSocket } from "../lib/ws-client";
import { ulid } from "ulidx";

const WS_URL =
  typeof window !== "undefined"
    ? import.meta.env.VITE_WS_URL ||
      `ws://${window.location.hostname}:8000/api/ws`
    : "ws://localhost:8000/api/ws";

const INITIAL_STATE = {
  connectionStatus: "disconnected",
  queryStatus: "idle",
  answer: "",
  sources: [],
  trust: null,
  consistency: null,
  error: null,
};

export function useWebSocketQuery() {
  const [state, setState] = useState(INITIAL_STATE);
  const wsRef = useRef(null);
  const currentQueryId = useRef(null);

  const handleMessage = useCallback((msg) => {
    // Filter stale message IDs — prevent race conditions on rapid queries
    if (msg.id && msg.id !== currentQueryId.current) return;

    switch (msg.type) {
      case "connected":
        setState((s) => ({ ...s, connectionStatus: "connected" }));
        break;
      case "status":
        setState((s) => ({ ...s, queryStatus: msg.stage }));
        break;
      case "sources":
        setState((s) => ({ ...s, sources: msg.sources }));
        break;
      case "token":
        setState((s) => ({ ...s, answer: s.answer + msg.content }));
        break;
      case "trust":
        setState((s) => ({
          ...s,
          trust: { score: msg.score, breakdown: msg.breakdown },
        }));
        break;
      case "consistency":
        setState((s) => ({
          ...s,
          consistency: {
            score: msg.score,
            rephrases_matched: msg.rephrases_matched,
          },
        }));
        break;
      case "done":
        setState((s) => ({ ...s, queryStatus: "done" }));
        break;
      case "cancelled":
        setState((s) => ({ ...s, queryStatus: "cancelled" }));
        break;
      case "error":
        setState((s) => ({
          ...s,
          queryStatus: "error",
          error: { code: msg.code, message: msg.message },
        }));
        break;
      default:
        break;
    }
  }, []);

  useEffect(() => {
    const ws = new TrustRAGWebSocket(WS_URL);
    wsRef.current = ws;
    ws.subscribe(handleMessage);
    ws.connect();
    return () => ws.close();
  }, [handleMessage]);

  const sendQuery = useCallback((text, options = {}) => {
    const id = ulid();
    currentQueryId.current = id;
    setState((s) => ({
      ...s,
      queryStatus: "retrieving",
      answer: "",
      sources: [],
      trust: null,
      consistency: null,
      error: null,
    }));
    wsRef.current.send({
      type: "query",
      id,
      text,
      top_k: options.top_k || 5,
      min_trust_score: options.min_trust_score || 0,
    });
  }, []);

  const cancelQuery = useCallback(() => {
    if (currentQueryId.current) {
      wsRef.current.send({ type: "cancel", id: currentQueryId.current });
    }
  }, []);

  const sendFeedback = useCallback((rating, comment = "") => {
    if (currentQueryId.current) {
      wsRef.current.send({
        type: "feedback",
        id: currentQueryId.current,
        rating,
        comment,
      });
    }
  }, []);

  return { ...state, sendQuery, cancelQuery, sendFeedback };
}
