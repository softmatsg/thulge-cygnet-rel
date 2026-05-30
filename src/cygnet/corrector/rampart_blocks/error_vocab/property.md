# Property-error refinement

A ``PropertyError`` means the query used a property with a type
that doesn't match its declaration. The structural validator
detected this at the AST stage by comparing the literal's inferred
type against the schema.

## Key fields

- ``property_name``: the property whose use is invalid.
- ``declared_type``: the type the schema declares (e.g.
  ``INTEGER``, ``STRING``, ``DATETIME``).
- ``used_type``: the literal type the query supplied. EXPLAIN-based
  detection currently fires only on ``STRING``-vs-numeric and a few
  other shapes; the AST backend is more sensitive.
- ``query_context``: surrounding fragment.
- ``did_you_mean``: populated only when the failure is actually a
  reference-not-found (a typo'd property name); empty for pure
  type-mismatch failures.

## Typical refinements

1. **Wrong literal type** — convert. ``WHERE s.year = "2020"`` →
   ``WHERE s.year = 2020`` when ``year`` is declared ``INTEGER``.
   ``WHERE s.created_at = "2026-01-01"`` →
   ``WHERE s.created_at = datetime("2026-01-01T00:00:00")`` when
   declared ``DATETIME``.
2. **Wrong property name (``did_you_mean`` populated)** —
   substitute as for ``SchemaError``.
3. **Property used in a comparison that doesn't match its type** —
   the conversion may need a Cypher type function: ``toInteger(...)``,
   ``toString(...)``, ``date(...)``, ``datetime(...)``.

Don't change the comparison operator or the semantic of the clause;
only the literal/type-coercion needs to move.
