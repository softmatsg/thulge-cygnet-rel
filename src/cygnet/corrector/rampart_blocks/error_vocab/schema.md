# Schema-error refinement

A ``SchemaError`` means the query referenced a label, relationship
type, or property that is not declared in the schema. The structural
validator has identified exactly which token is unknown and what
kind of reference it is.

## Key fields

- ``unknown_reference``: the token in the query that does not match
  any declared name.
- ``reference_kind``: one of ``label``, ``relationship``, or
  ``property``. Drives where to look for the right name.
- ``did_you_mean``: pre-ranked close matches over the relevant
  vocabulary, scored by edit distance. **Prefer these** — they
  are pre-filtered to be plausible substitutions.
- ``available_in_scope``: the full option space at the failure
  site. For a label miss this is every label in the schema; for a
  property miss this is just properties on the bound variable's
  label(s). Use this when ``did_you_mean`` is empty or none of its
  suggestions are obviously right.
- ``available_in_scope_truncated``: ``True`` means the vocabulary
  exceeded the per-payload cap; the surfaced list is a prefix only.

## Typical refinements

1. **``did_you_mean`` has one strong match** — substitute the first
   suggestion for ``unknown_reference`` and return.
2. **``did_you_mean`` is empty but ``available_in_scope`` is
   short** — pick the most semantically plausible label/rel/property
   from the scope. The query's other tokens give hints (e.g.,
   ``s.created_at`` constrains the label to something with a
   ``created_at`` property).
3. **``available_in_scope_truncated`` is True** — the schema is
   large; ``did_you_mean`` is now the only structured hint. If
   ``did_you_mean`` is also empty, return an empty block to abort.
4. **Property miss with multi-label binding** — check that the
   property is declared on at least one of the variable's labels.
   If not, the user may have bound the variable to the wrong label.

Keep the rest of the query intact. Schema fixes are local
substitutions, not rewrites.
