# Parse-error refinement

A ``ParseError`` means the query did not parse against Neo4j's Cypher
grammar. The structural validator chain has surfaced the offending
position before any schema or cost checks ran, so the schema is not
the problem.

## Key fields

- ``message``: the parser's error text. Often points at the
  immediate token; the actual cause may be a few characters earlier
  (an unbalanced bracket or an unterminated string).
- ``line``, ``column``: 1-based position.
- ``excerpt_with_caret``: two-line excerpt with ``^`` under the
  offending column. This is the highest-signal field for parse
  errors — read it first.
- ``snippet``: a short context fragment.

## Typical refinements

1. **Unbalanced brackets** — add the missing ``)``, ``]``, or ``}``.
   The caret may point past the end of the relevant clause; scan
   leftward for the unmatched opener.
2. **Unterminated string literals** — close the quote.
3. **Typo'd clause keywords** — ``MTACH`` → ``MATCH``, ``RETRUN`` →
   ``RETURN``. The error message often contains the suggestion.
4. **Reserved word as identifier** — escape with backticks or
   rename. ``MATCH (n:Order)`` works; ``Order`` is reserved in some
   contexts.
5. **``CALL { ... }`` subqueries** — the validator's AST backend
   does not parse Neo4j 4.0+ subqueries; rewrite using ``WITH`` and
   chained ``MATCH``.

Return the refined query in a ``cypher`` block. If the parse error
is malformed beyond minimal fixing, return an empty block to abort.
