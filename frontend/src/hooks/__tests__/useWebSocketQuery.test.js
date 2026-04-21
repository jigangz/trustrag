import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useWebSocketQuery } from "../useWebSocketQuery";

// Mock ws-client module
let mockSubscribeCallback = null;
const mockSend = vi.fn();
const mockConnect = vi.fn();
const mockClose = vi.fn();

vi.mock("../../lib/ws-client", () => ({
  TrustRAGWebSocket: class MockTrustRAGWebSocket {
    constructor() {
      // expose instance methods
    }
    subscribe(fn) {
      mockSubscribeCallback = fn;
      return () => {};
    }
    connect() {
      mockConnect();
    }
    close() {
      mockClose();
    }
    send(msg) {
      mockSend(msg);
    }
  },
}));

// Mock ulidx to return predictable IDs
let ulidCounter = 0;
vi.mock("ulidx", () => ({
  ulid: () => `TEST-ID-${++ulidCounter}`,
}));

describe("useWebSocketQuery", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSubscribeCallback = null;
    ulidCounter = 0;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("initial state is idle and disconnected", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    expect(result.current.connectionStatus).toBe("disconnected");
    expect(result.current.queryStatus).toBe("idle");
    expect(result.current.answer).toBe("");
    expect(result.current.sources).toEqual([]);
    expect(result.current.trust).toBeNull();
    expect(result.current.consistency).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("connected message updates connection status", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    act(() => {
      mockSubscribeCallback({ type: "connected", server_version: "0.2.0" });
    });

    expect(result.current.connectionStatus).toBe("connected");
  });

  it("token messages accumulate answer text", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    // Start a query first so currentQueryId is set
    act(() => {
      result.current.sendQuery("test question");
    });

    const queryId = "TEST-ID-1";

    act(() => {
      mockSubscribeCallback({ type: "token", id: queryId, content: "Hello" });
    });
    act(() => {
      mockSubscribeCallback({ type: "token", id: queryId, content: " world" });
    });

    expect(result.current.answer).toBe("Hello world");
  });

  it("cancel sends cancel message with current query ID", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    act(() => {
      result.current.sendQuery("test question");
    });

    act(() => {
      result.current.cancelQuery();
    });

    expect(mockSend).toHaveBeenCalledWith({
      type: "cancel",
      id: "TEST-ID-1",
    });
  });

  it("stale message IDs are ignored", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    // Send first query
    act(() => {
      result.current.sendQuery("first question");
    });
    const firstId = "TEST-ID-1";

    // Send some tokens for first query
    act(() => {
      mockSubscribeCallback({ type: "token", id: firstId, content: "old " });
    });

    // Send second query (resets state, new ID)
    act(() => {
      result.current.sendQuery("second question");
    });

    // Token from first query arrives late — should be ignored
    act(() => {
      mockSubscribeCallback({
        type: "token",
        id: firstId,
        content: "stale",
      });
    });

    // Answer should be empty (reset by second sendQuery, stale token ignored)
    expect(result.current.answer).toBe("");

    // Token from second query arrives — should be accepted
    const secondId = "TEST-ID-2";
    act(() => {
      mockSubscribeCallback({
        type: "token",
        id: secondId,
        content: "fresh",
      });
    });
    expect(result.current.answer).toBe("fresh");
  });

  it("sendQuery resets state and sends query message", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    act(() => {
      result.current.sendQuery("What is fall protection?");
    });

    expect(result.current.queryStatus).toBe("retrieving");
    expect(result.current.answer).toBe("");
    expect(result.current.sources).toEqual([]);
    expect(result.current.trust).toBeNull();
    expect(result.current.error).toBeNull();

    expect(mockSend).toHaveBeenCalledWith({
      type: "query",
      id: "TEST-ID-1",
      text: "What is fall protection?",
      top_k: 5,
      min_trust_score: 0,
    });
  });

  it("status message updates queryStatus to the stage", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    act(() => {
      result.current.sendQuery("test");
    });

    act(() => {
      mockSubscribeCallback({
        type: "status",
        id: "TEST-ID-1",
        stage: "generating",
      });
    });

    expect(result.current.queryStatus).toBe("generating");
  });

  it("trust message sets trust score and breakdown", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    act(() => {
      result.current.sendQuery("test");
    });

    const breakdown = { source_quality: 85, answer_relevance: 90 };
    act(() => {
      mockSubscribeCallback({
        type: "trust",
        id: "TEST-ID-1",
        score: 87,
        breakdown,
      });
    });

    expect(result.current.trust).toEqual({ score: 87, breakdown });
  });

  it("error message sets error state", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    act(() => {
      result.current.sendQuery("test");
    });

    act(() => {
      mockSubscribeCallback({
        type: "error",
        id: "TEST-ID-1",
        code: "GROQ_RATE_LIMIT",
        message: "Rate limited",
      });
    });

    expect(result.current.queryStatus).toBe("error");
    expect(result.current.error).toEqual({
      code: "GROQ_RATE_LIMIT",
      message: "Rate limited",
    });
  });

  it("sendFeedback sends feedback message", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    act(() => {
      result.current.sendQuery("test");
    });

    act(() => {
      result.current.sendFeedback("good", "Very helpful!");
    });

    expect(mockSend).toHaveBeenCalledWith({
      type: "feedback",
      id: "TEST-ID-1",
      rating: "good",
      comment: "Very helpful!",
    });
  });

  it("done message sets queryStatus to done", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    act(() => {
      result.current.sendQuery("test");
    });

    act(() => {
      mockSubscribeCallback({
        type: "done",
        id: "TEST-ID-1",
        audit_id: "audit-123",
      });
    });

    expect(result.current.queryStatus).toBe("done");
  });

  it("cancelled message sets queryStatus to cancelled", () => {
    const { result } = renderHook(() => useWebSocketQuery());

    act(() => {
      result.current.sendQuery("test");
    });

    act(() => {
      mockSubscribeCallback({ type: "cancelled", id: "TEST-ID-1" });
    });

    expect(result.current.queryStatus).toBe("cancelled");
  });
});
