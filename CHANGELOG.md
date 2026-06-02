# Changelog

All notable changes to CYGNET. The project pre-dates a stable
public API; semantic versioning applies once the library reaches
v1.0. Breaking changes during the v0.0.x series are expected and
called out explicitly per release.

## v0.0.46 — Rel-endpoint compatibility check in the AST backend

The AST backend gains a new structural check that verifies path
patterns of the form ``(a:X)-[:R]->(b:Y)`` reference relationship
types whose declared source/target labels are compatible with the
labels bound on each end. Catches plausible-mistake swap errors
that previously slipped through every backend in the chain — for
example, ``MATCH (a:Customer)-[:CONTAINS]->(b:Product)`` against a
schema where ``CONTAINS`` is declared to go ``Order -> Product``.

### Library

- ``cygnet.validator.ASTValidator`` now performs a rel-endpoint
  compatibility check after its existing schema-name checks. For
  each path segment in the parsed query, if the rel type's declared
  source/target labels do not match the labels bound on each end
  (direction-aware), the backend emits a ``cygnet.SchemaError``
  with ``reference_kind="relationship"`` and ``did_you_mean``
  populated from the rel types that actually connect those labels.

  Edge cases handled:

  * **Missing label on either end** — skip the check; the caller may
    have intended any label.
  * **No rel type specified** (``-[]->``) — skip; any rel type
    could apply.
  * **Multi-rel-type disjunction** (``-[:R1|R2]->``) — accepted
    if any one rel type is compatible.
  * **Multi-label conjunction on nodes** (``(a:X:Y)``) — accepted
    if any bound label matches the declared endpoint.
  * **Variable-length paths** (``-[:R*1..3]->``) — checked at the
    bracket's rel type against the path's endpoints.
  * **Undirected patterns** (``-[:R]-``) — accepted if R connects
    the labels in either direction.

- Error vocabulary, mirror construction, schema model, and the
  three other validator backends are unchanged.

### Measurement (informational)

The nine-schema validator-quality evaluation shipped in the
accompanying paper was re-run after the check landed. Chain mean
TPR rose from 0.548 to 0.648 across 9 schemas; FPR stayed at
0.000 across every schema and every backend. Schema-category
recall roughly doubled. Parse, property, and constraint
categories are unchanged.

## v0.0.45 — Initial public release

The first publicly available release of CYGNET. The library
provides a programmable gate between a Cypher-generating system
and a Neo4j graph database: structural validation across four
backends (regex, ANTLR-AST, EXPLAIN, mirror-execute), cost
gating via EXPLAIN planner output, and a corrector loop with
five pluggable LLM-backed implementations.

See ``README.md`` for installation and quick-start examples.
See ``KNOWN_ISSUES.md`` for documented limitations and follow-ups.
