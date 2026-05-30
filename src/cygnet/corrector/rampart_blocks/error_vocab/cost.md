# Cost-error refinement

A ``CostError`` means the query passed structural validation but its
EXPLAIN plan exceeds the configured row or db-hit threshold. The
plan would run on production, just at unacceptable cost.

## Key fields

- ``estimated_rows``: the planner's total row estimate.
- ``estimated_dbhits``: summed-operator row proxy (EXPLAIN does not
  surface real db-hits).
- ``threshold_used``: the configured row threshold.
- ``cost_driver``: the operator most responsible, formatted as
  ``<OperatorName> on [<identifiers>]``.
- ``suggested_mitigations``: pre-computed strings keyed to the cost
  driver category (cartesian, var-length, all-nodes scan, label
  scan). Read these first.
- ``estimated_cost_breakdown``: top-5 operators by row estimate.
  Tells you which variables/labels are blowing up the cost.

## Primary refinement strategies

1. **Add a ``LIMIT`` clause.** The single most effective refinement
   for most cost rejections, especially when the user's intent is
   exploratory ("show me some samples"). Always safe; never changes
   semantics for already-bounded queries.

2. **Add a ``WHERE`` filter on a property covered by an index.**
   Look at ``estimated_cost_breakdown``: if the top operator is a
   ``NodeByLabelScan`` or ``AllNodesScan``, a ``WHERE`` on an
   indexed property (often the schema's uniqueness-constraint
   property — ``s.id = $id`` is the canonical pattern) collapses
   the scan to a lookup.

3. **Eliminate cartesian products.** When ``cost_driver`` is
   ``CartesianProduct``, the two MATCH patterns are disconnected.
   Introduce a relationship between them (``MATCH (s:Sample)-
   [r:MEASURED_BY]->(m:Measurement) ...``) rather than the comma-
   separated pattern.

4. **Bound variable-length expansion.** When ``cost_driver`` is
   ``VarLengthExpand``, replace ``-[*]->`` with ``-[*..3]->`` (or a
   relationship-type filter ``-[:REL*..3]->``). The exact depth
   bound depends on the user's intent; pick the smallest that
   plausibly preserves the query's meaning.

5. **Add a relationship-type filter.** When ``cost_driver`` mentions
   ``Expand(All)``, restricting the relationship type narrows the
   expansion.

Prefer the smallest mitigation that addresses the cost driver.
``LIMIT 100`` is almost always the right first move; reach for
filters and pattern restructuring only when ``LIMIT`` alone is
inappropriate for the query's intent.
