# Pattern: WITH...WHERE staging

Use ``WITH`` to project intermediate values, then ``WHERE`` to
filter on them. Lets you filter on aggregations or computed
properties that can't appear in a plain ``WHERE`` clause attached
to ``MATCH``.

```cypher
MATCH (s:Sample)-[:MEASURED_BY]->(m:Measurement)
WITH s, count(m) AS measurement_count
WHERE measurement_count > 3
RETURN s.id, measurement_count
```

```cypher
MATCH (s:Sample)
WITH s, date() - date(s.created_at) AS age_days
WHERE age_days < 30
RETURN s.id, age_days
```

``WITH`` also lets you break a query into stages, each of which the
planner optimises independently. When the cost gate has flagged a
giant join, splitting via ``WITH`` (with a ``LIMIT`` on the first
stage) often improves the plan without changing the result.

Variables not carried through ``WITH`` are dropped from scope —
this is a common source of "unknown variable" parse errors after a
refinement.
