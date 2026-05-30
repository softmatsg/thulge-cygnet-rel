# CYGNET

**Cypher Gate for Neural Execution Triage.**

CYGNET is a Python library that gates LLM-generated Cypher queries before
they execute against production Neo4j. It does three jobs:

1. **Validate** — pre-execution structural checks across parse, schema,
   property, and constraint categories, with a discriminated error
   vocabulary designed for agent refinement loops.
2. **Estimate cost** — execution-plan analysis via Neo4j's `EXPLAIN`
   without running the query, to catch unbounded-scan and Cartesian-
   product mistakes early.
3. **Correct** — RAMPART-backed structural correction that turns a
   validator failure into a refined query, composed with protocol-retry
   and refinement-loop decorators that match the production
   configuration.

## Installation

```bash
pip install thulge-cygnet
```

Extras for optional components:

- `pip install thulge-cygnet[ast]` — ANTLR4-based syntax validator
- `pip install thulge-cygnet[corrector]` — LLM-backed correction
  (Anthropic, OpenAI, Google Gemini, Ollama)
- `pip install thulge-cygnet[mcp]` — Model Context Protocol server
- `pip install thulge-cygnet[http]` — HTTP transport (FastAPI + uvicorn)

## Quick start — validate

```python
from cygnet import Gate, GateConfig

gate = Gate.from_config(GateConfig.model_validate({
    "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "..."},
    "schema": {"source": "spec_file", "spec_path": "schema.yaml"},
    "validator": {"backends": ["builtin", "ast", "explain"]},
}))

result = gate.chain.validate("MATCH (n:Moive) RETURN n.title")
print(result.passed)          # False
print(result.failed_stage)    # "schema"
print(result.all_errors[0])   # SchemaError(unknown_reference='Moive', did_you_mean=['Movie'])
```

## Quick start — correct

```python
from cygnet import GateError, run_correction, RampartCorrector
from cygnet.corrector.llm import GoogleGeminiClient

corrector = RampartCorrector(
    GoogleGeminiClient(model="gemini-2.5-flash"),
    model="gemini-2.5-flash", provider="google",
)

payload = result.all_errors[0]
refined = run_correction(
    corrector=corrector,
    query="MATCH (n:Moive) RETURN n.title",
    error_context=GateError(category=payload.category, payload=payload),
    validator_chain=gate.chain,
    schema=gate.get_schema(),
)
print(refined.action)         # "refined"
print(refined.refined_query)  # "MATCH (n:Movie) RETURN n.title"
```

## Quick start — mirror

A schema mirror is a sparse synthetic Neo4j instance that has every
declared label, relationship type, property, constraint, and index but
only one instance of each. It lets `EXPLAIN`-based validation work
without touching the production graph.

```python
from cygnet.mirror import MirrorGraphBuilder
from cygnet.schema import load_schema_spec, validate_mirror

schema = load_schema_spec("schema.yaml")
MirrorGraphBuilder(driver, database="neo4j").build_from_schema(schema)
report = validate_mirror(schema, driver, database="neo4j",
                         property_sample_limit=10000)
assert report.ok
```

## Requirements

- Python 3.11 or newer
- Neo4j 5.x (the library uses `RANGE` indexes; Neo4j 4 `BTREE` indexes
  are normalised on read)

## Documentation

See module-level docstrings throughout `cygnet.*`. The public API surface
is the `cygnet` package root — everything re-exported there is part of
the supported interface.

## Known issues

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for documented limitations and
follow-ups (mirror property-sampling defaults, Gemma JSON-mode on the
correction path, EXPLAIN notification handling).

## Paper

CYGNET accompanies the paper *Cypher Gate for Neural Execution Triage*
(citation forthcoming — DOI / arXiv link will be added on publication).

```bibtex
@misc{cygnet2026,
  author       = {Nikodem Tomczak},
  title        = {CYGNET: Cypher Gate for Neural Execution Triage},
  year         = {2026},
  howpublished = {Thulge Labs},
}
```

## Licence

See [LICENSE](LICENSE). Free for non-commercial use including academic
research, personal projects, and evaluation. Commercial production use
requires a separate commercial licence from Thulge Labs.

Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.
