# Pattern: OPTIONAL MATCH

Return rows that may or may not have a related node, treating
missing relationships as ``NULL`` rather than filtering the row
out. The canonical fix for empty-result rejections caused by an
overly-strict inner ``MATCH``.

```cypher
MATCH (s:Sample)
OPTIONAL MATCH (s)-[:MEASURED_BY]->(m:Measurement)
RETURN s, m
```

```cypher
MATCH (s:Sample {id: $id})
OPTIONAL MATCH (s)-[:OF_COMPOUND]->(c:Compound)
RETURN s.id, c.formula
```

``OPTIONAL MATCH`` always succeeds — when there's no match, the
bound variable is ``NULL``. Use ``coalesce(x.prop, default)`` in the
``RETURN`` clause when you need a placeholder rather than ``NULL``.

Don't use ``OPTIONAL MATCH`` for cost-driven refinements; it
typically *increases* the planner's row estimate.
