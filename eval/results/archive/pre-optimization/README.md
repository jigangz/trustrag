# Pre-Optimization Benchmark Results (Archived)

These results were generated **before** the v2-completion optimizations
(embedding cleanup, merged prompt, query cache). Numbers may reflect:

- Partial runs interrupted by Groq TPD limits
- Ralph Loop auto-generated placeholders
- Hallucination check via separate 2-call path (not merged-prompt)
- RAGAS judge on older/default provider (not Gemini Flash Lite)

**Canonical v1.0.0 benchmark results** live in `eval/results/` root with
names like `2026-04-23-semantic-15q.json` and `2026-04-23-hybrid-15q.json`.

Methodology for the canonical runs is in `docs/releases/v0.3.0-hybrid.md`.
See also `docs/superpowers/specs/2026-04-21-trustrag-v2-completion-design.md`
§5 for the v2-completion eval design (Gemini 2.5 Flash Lite judge,
`max_workers=1` to survive free-tier RPM, `nocache=True` to bypass cache
for benchmark fidelity per SIGN-111).

## Files

- `2026-04-20-*.json` — Ralph Loop iteration outputs, partial/projected numbers
- `2026-04-20-comparison.md` — Ralph-generated comparison doc
- `2026-04-21-production-hybrid.json` — real Railway hybrid run, pre-WS1 optimization (30-60s/query, 2-call hallucination)
- `2026-04-21-real-hybrid.json` — local docker hybrid run, pre-WS1

Do not cite these numbers in README or release notes. They're retained
only for version history and to show evolution of methodology.
