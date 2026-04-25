# MCP Server Verification — trustrag-mcp 0.1.2

**Verified**: 2026-04-25
**Backend**: `https://trustrag-production.up.railway.app` (Railway production)
**MCP package**: [`trustrag-mcp 0.1.2`](https://pypi.org/project/trustrag-mcp/0.1.2/)
**Client**: Claude Code (via `mcp__trustrag__*` tool surface — same `trustrag-mcp.exe` stdio transport that Claude Desktop uses)

This document captures real input/output from each of the 3 tools the
MCP server exposes, hit live against production. Screenshots from the
Claude Desktop UI may be added later as visual companions; this log is
the canonical end-to-end verification for v0.5.0-mcp.

---

## Tool 1 — `trustrag_query`

**Input:**

```json
{
  "question": "What fall protection does OSHA require at 6 feet?"
}
```

**Output:**

```
**Answer** (Trust: 72.8/100):

According to the sources, OSHA requires fall protection for employees on a
scaffold more than 10 feet above a lower level [Source: OSHA3150.pdf, page 41].
There is no specific requirement mentioned for 6 feet. The standard requires
fall protection for employees more than 10 feet above a lower level, but does
not provide information on requirements at 6 feet [Source: OSHA3150.pdf,
page 5].

**Sources**:
- OSHA3150.pdf (page 41, similarity 0.75)
- OSHA3150.pdf (page 41, similarity 0.75)
- OSHA3150.pdf (page 5, similarity 0.74)
- OSHA3150.pdf (page 5, similarity 0.74)
- OSHA3150.pdf (page 13, similarity 0.73)

**Trust Breakdown**: {
  'agreement': 18.1,
  'retrieval': 74.2,
  'source_count': 15.0,
  'hallucination': 10.0
}
```

**Notes:**
- Trust score 72.8 reflects partial coverage — the corpus contains the 10-ft
  scaffold rule but not a generic 6-ft requirement. The model correctly
  surfaces what the corpus *does* say rather than confabulating a 6-ft
  answer. This is the trust system working as intended.
- 4-factor trust breakdown shows source agreement is the weakest signal
  (18.1) because two answers point to different page ranges. Retrieval and
  hallucination checks are strong.

---

## Tool 2 — `trustrag_upload_document`

**Input:**

```json
{
  "file_path": "C:/tmp/test-upload.pdf"
}
```

(Test fixture: a single-page OSHA fact sheet covering hearing conservation,
confined spaces, and recordkeeping. Generated locally for this demo.)

**Output:**

```
Uploaded: document_id=b7106d01-9f81-4733-8132-8d7ef13a97e5, 1 chunks indexed
```

**Notes:**
- End-to-end flow: PDF → multipart POST to `/api/documents/upload` →
  backend parses + chunks + embeds with FastEmbed → row inserted in
  Postgres + pgvector.
- Returned `document_id` is a UUID; `total_chunks=1` because the test PDF
  is short. Larger PDFs (e.g. the full OSHA3150 manual) typically chunk
  to 50-200 pieces.

---

## Tool 3 — `trustrag_get_audit_log`

**Input:**

```json
{
  "limit": 5,
  "max_trust_score": 90
}
```

(`max_trust_score` filtering is applied client-side in
[`server.py`](../packages/trustrag-mcp/src/trustrag_mcp/server.py) — the
backend's `/api/audit/` only takes `limit`/`offset`; the MCP wrapper
fetches a wider window and trims. See v0.1.2 changelog in
[`docs/releases/v0.5.0-mcp.md`](releases/v0.5.0-mcp.md).)

**Output (truncated for brevity, formatted):**

```
**[unknown time] Trust: 83.2**
Q: What is the minimum height threshold for fall protection in construction?
A: The minimum height threshold for fall protection in construction is 10 feet
(3.1 m) above a lower level [Source: OSHA3150.pdf, p. 41]...

**[unknown time] Trust: 62.0**
Q: fall protection
A: Based on the provided sources, here are the answers to the question
"fall protection": 1. What types of scaffolds require fall protection?...

**[unknown time] Trust: 62.0**
Q: fall protection
A: Based on the provided source documents...

**[unknown time] Trust: 62.0**
Q: fall protection
A: Based on the provided source documents...

**[unknown time] Trust: 79.6**
Q: OSHA Form 300A
A: OSHA Form 300A is a form used by employers to record and report
work-related injuries and illnesses...
```

**Notes:**
- Filter worked: every returned entry has trust < 90.
- `[unknown time]` placeholder appears because the backend currently
  serializes `created_at` as `null` in the audit response. This is a
  separate backend issue (not in the MCP wrapper); tracked for a
  follow-up patch. The MCP wrapper handles missing timestamps
  gracefully so the tool still returns meaningful results.

---

## Summary

| Tool                          | Status | Real backend hit | Trust system surfaced |
|-------------------------------|--------|------------------|-----------------------|
| `trustrag_query`              | ✅     | Yes              | Yes (72.8/100, 4-factor breakdown) |
| `trustrag_upload_document`    | ✅     | Yes              | N/A (upload, no scoring) |
| `trustrag_get_audit_log`      | ✅     | Yes              | Yes (per-entry confidence_score) |

All 3 tools work end-to-end against production with `trustrag-mcp 0.1.2`.
This closes **P5-GATE** from the v0.2 design spec.

To reproduce: install `trustrag-mcp>=0.1.2`, configure `claude_desktop_config.json`
per [`docs/releases/v0.5.0-mcp.md`](releases/v0.5.0-mcp.md), restart
Claude Desktop, ask any of the prompts above.
