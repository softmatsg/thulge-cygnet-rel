# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Cost gate: pre-execution cost estimation via Neo4j EXPLAIN.

Sends ``EXPLAIN <query>`` to the configured Neo4j (typically the mirror),
parses the plan, identifies the most actionable operator above the
threshold, and returns a ``CostGateResult`` with row/db-hit estimates
and tailored mitigation suggestions.

The cost gate is *pre*-execution: it does not run the query. PROFILE
would give real db-hits but defeats the purpose, so we use EXPLAIN and
proxy db-hits via summed operator row estimates.

Modules:

- ``explain_parser``: turns Neo4j's nested plan dict into a flat
  :class:`ExplainPlan`.
- ``gate``: the :class:`CostGate` class itself.
- ``estimator`` (planned): pluggable cost-estimator interface for users
  who want an ML-backed estimator instead of EXPLAIN.
"""

from cygnet.cost.explain_parser import parse_explain_plan
from cygnet.cost.gate import CostGate
from cygnet.models import ExplainOperator, ExplainPlan

__all__ = [
    "CostGate",
    "ExplainOperator",
    "ExplainPlan",
    "parse_explain_plan",
]
