# Constraint-error refinement

A ``ConstraintError`` means a ``CREATE`` or ``MERGE`` pattern would
violate a schema constraint — most commonly a required property
missing from a node pattern, or a literal value that would collide
with a uniqueness constraint.

## Key fields

- ``constraint_id``: the constraint's identifier (e.g.
  ``sample_id_unique``, ``compound_formula_required``).
- ``constraint_kind``: one of ``uniqueness``, ``existence``,
  ``type``.
- ``property_name``: the property the constraint references.
- ``violating_value``: the literal value that would write
  (uniqueness only, when statically detectable). ``None`` for
  existence violations and for runtime-only detection paths.

## Typical refinements

1. **Existence violation (missing required property)** — add the
   property to the node's property map. ``CREATE (s:Sample
   {created_at: datetime()})`` →
   ``CREATE (s:Sample {id: $id, created_at: datetime()})`` when
   ``id`` is required. The user must supply the value; introduce a
   parameter (``$id``) rather than inventing one.
2. **Uniqueness violation (literal collision)** — change the
   literal value to one the user has not used elsewhere, or
   convert ``CREATE`` to ``MERGE`` so the pattern matches the
   existing row instead of inserting a duplicate.
3. **Type constraint** — coerce the literal to the declared type,
   same as ``PropertyError``.

For uniqueness violations where the violating value comes from a
parameter, the only safe refinement is ``MERGE``. Do not invent
values — return an empty block to abort if no parameter-based fix
is possible.
