# Pattern: relationship traversal

Connect two labels via a typed relationship. The canonical fix for
cartesian-product cost rejections.

```cypher
MATCH (s:Sample)-[:MEASURED_BY]->(m:Measurement)
RETURN s, m
```

```cypher
MATCH (s:Sample)-[r:MEASURED_BY]->(m:Measurement)
WHERE r.instrument = 'spectrometer'
RETURN s.id, m.temperature
```

Direction matters in Cypher — ``-[r]->`` and ``<-[r]-`` produce
different result sets when the schema's edge is directional. Match
the direction declared in ``RelationshipType``'s
``source_label``/``target_label``.

Alternation ``[:A|B]`` lets a single pattern cover multiple types;
each type is checked independently against the schema vocabulary.
