# Pattern: aggregation

Count or aggregate rows by a property; canonical shape for analytics
queries against a labelled set.

```cypher
MATCH (s:Sample)
RETURN count(s) AS sample_count
```

```cypher
MATCH (s:Sample)
RETURN s.created_at.year AS year, count(*) AS samples
ORDER BY year DESC
```

```cypher
MATCH (m:Measurement)
RETURN avg(m.temperature) AS mean_temp, count(m) AS n
```

When the cost gate has flagged the underlying scan, the aggregation
itself usually isn't the problem — narrow the ``MATCH`` with a
``WHERE`` before aggregating rather than ``LIMIT``-ing the result of
the aggregation (which truncates the answer rather than the scan).
