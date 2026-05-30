# CYGNET — Known Issues

Documented limitations and follow-ups in the current release. Each
entry describes the behaviour, when it bites, and the recommended
workaround.

---

## 1. `validate_mirror` property-sample default is too low for sparse schemas

**Symptom.** `validate_mirror(declared, driver)` reports a declared
property as missing when the property is genuinely present but only
populated on a small fraction of nodes.

**Cause.** The default `property_sample_limit` is `50`. The check
samples up to that many nodes per label; if the property is sparser
than ~1/N for that sample, no instance is observed and the property is
reported missing.

**Workaround.** Pass an explicit higher limit when you know the schema
has sparse properties:

```python
report = validate_mirror(
    declared, driver, database="neo4j",
    property_sample_limit=10000,
)
```

**Status.** The default will be raised in a future release, likely
together with an auto-scale heuristic based on per-label node count.
Callers that need deterministic behaviour today should pass an explicit
value.

---

## 2. Gemma JSON-mode protocol failures on the correction path

**Symptom.** When the corrector is configured with a Gemma model
(e.g. `gemma-4-31b-it`, `gemma-4-26b-a4b-it`) and the corrector is
asked to return a JSON object via the `_CYPHER_ONLY_RESPONSE_SCHEMA`,
the model frequently returns text that fails the parser as
`ProtocolMalformed: invalid JSON`. The protocol-retry decorator
(`ProtocolRetryingCorrector`) recovers some of these but on the harder
schemas the success rate of recovery is materially below that of
Gemini and other providers.

**Cause.** Gemma's structured-output adherence under the
`response_mime_type="application/json"` contract is weaker than the
sibling Gemini models. The output sometimes contains a leading prose
preface, a trailing prose explanation, or two concatenated JSON
objects, none of which the strict parser will accept.

**Workaround.** Two options:

- **Disable JSON mode** for Gemma cells via
  `GoogleGeminiClient(model=..., response_schema=None)` and rely on the
  parser's fence-aware fallback. This trades structured-output
  enforcement for higher tolerance.
- **Keep JSON mode**, accept the protocol failures as part of the
  baseline error rate for Gemma. The
  `ProtocolRetryingCorrector(retries=2)` default already gives 3 total
  attempts, which masks most of the noise.

**Status.** A provider-aware default that turns off
`response_schema` for Gemma identifiers is on the follow-up list.
Until then, configure explicitly.

---

## 3. EXPLAIN backend does not escalate Neo4j notifications to failures

**Symptom.** Queries that reference an unknown label or relationship
type (e.g. `MATCH (n:Moive) RETURN n`) often pass the `explain`
validator backend even though Neo4j returns `01N50` (unknown label) or
`01N51` (unknown relationship type) notifications. These appear as
warnings on the result, not as the result-rejecting verdict the
upstream `builtin` and `mirror_execute` backends produce.

**Cause.** The `explain` backend is intentionally permissive: its
remit is *cost-plan* analysis, not schema validation. Neo4j's
notifications are advisory and would in principle add a useful signal
here, but escalating them to failures is a design decision that the
current release defers.

**Impact.** Schemas where labels and relationship types accumulate
typos that are valid Cypher (i.e. structurally well-formed but
semantically wrong) get caught by `builtin` and the upstream chain
combination, but not by `explain` alone. If your validator
configuration uses only `["explain"]`, you will under-detect schema
errors.

**Recommended configuration.** Use the full chain
`["builtin", "ast", "explain", "mirror_execute"]`, which is the
default for `Gate.from_config(...)`. The chain catches schema and
property errors via `builtin` and constraint violations via
`mirror_execute`; `explain` then adds cost-plan signal on top.

**Status.** Notification-to-error escalation is on the follow-up list
for a future minor version, behind a configuration flag so existing
callers do not see behaviour changes.

---

Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.
