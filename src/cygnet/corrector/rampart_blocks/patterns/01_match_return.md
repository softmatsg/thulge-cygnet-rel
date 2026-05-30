# Pattern: simple MATCH...RETURN

The canonical lookup against a single label, returning either the
whole node or selected properties.

```cypher
MATCH (s:Sample)
RETURN s
```

```cypher
MATCH (s:Sample)
RETURN s.id, s.created_at
LIMIT 100
```

Add a ``LIMIT`` when the user's intent is exploratory or when the
cost gate has flagged an unbounded scan.
