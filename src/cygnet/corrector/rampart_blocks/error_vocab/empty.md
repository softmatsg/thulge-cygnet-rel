# Empty-result refinement

An ``EmptyResultError`` means the query parsed cleanly, validated
against the schema, passed the cost gate, and ran against
production — but returned zero rows (or, when an expected range is
configured, an out-of-range row count).

## Status

Empty-result detection is currently deferred in CYGNET; this block
exists so the corrector has guidance available when post-execution
detection lands. If you receive an ``EmptyResultError`` today, the
caller has wired in their own detection.

## Key fields

- ``query``: the query that executed.
- ``expected_range``: optional ``(min_rows, max_rows)`` configured
  by the caller. ``None`` when only the "zero rows" case is being
  detected.

## Typical refinements

1. **Over-constrained ``WHERE`` clause** — relax the filter.
   ``WHERE s.created_at > datetime("2026-01-01")`` may have no
   matches; consider broadening the range or dropping the clause.
2. **Wrong relationship direction** — flip ``->`` and ``<-`` if
   the schema's relationship semantics differ from what the query
   assumes.
3. **Typo'd literal value** — case sensitivity, leading/trailing
   whitespace, alternate spellings.
4. **Missing ``OPTIONAL MATCH``** — when the intent is "find these
   nodes and optionally their related items", an inner ``MATCH``
   filters out nodes without the relationship. Switch to
   ``OPTIONAL MATCH``.

Empty results don't carry structured hints the way other errors do;
the refinement is largely guesswork. If the query has no obvious
flaw, return an empty block to abort — the user should clarify
intent rather than receive an arbitrarily relaxed query.
