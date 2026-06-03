# Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

"""Dev-time Neo4j container lifecycle management.

Wraps the bundled :file:`docker-compose.yml` so library tests and
demo notebooks can start a fresh Neo4j 5 + APOC instance, load a
dataset, exercise the gate, and tear down — all from a single
``with Phase1Neo4j()`` block. Kept as a public utility so external
callers don't need to vendor it.

Designed to coexist with the integration-test container in
``tests/integration/`` (port 7688): this container binds bolt to
port 7689 and has its own named volumes for persistence across
runs. Both can run in parallel during dev.

The bundled ``docker-compose.yml`` ships as package data so the
class works from any installed location. The ``docker`` CLI is
invoked via :mod:`subprocess`; if Docker is not installed,
:meth:`Phase1Neo4j.start` raises a clear error from the underlying
``subprocess.run`` call.

The library-side ``cygnet.dev`` namespace is documented as
dev-time scaffolding, NOT part of the production API surface
— see :mod:`cygnet.dev`'s docstring.

``load_dataset`` accepts a callable so callers can plug in their
own loaders without entangling :mod:`cygnet.dev` with downstream
dataset code.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ServiceUnavailable

if TYPE_CHECKING:
    from cygnet.models import Schema

__all__ = [
    "PHASE1_AUTH",
    "PHASE1_BOLT_URI",
    "PHASE1_DATABASE",
    "DatasetLoader",
    "Phase1Neo4j",
]


class DatasetLoader(Protocol):
    """Signature of a dataset loader consumed by
    :meth:`Phase1Neo4j.load_dataset`.

    A loader is a callable taking the live Neo4j driver and the
    target database (as a keyword-only argument) and returning the
    post-load node count."""

    def __call__(self, driver: Driver, *, database: str = "neo4j") -> int: ...


logger = logging.getLogger("cygnet.dev.neo4j_lifecycle")


PHASE1_BOLT_URI: Final[str] = "bolt://127.0.0.1:7689"
"""Bolt URI of the dev Neo4j container. Port 7689 is reserved for
this dev-time container; the integration-test container at
``tests/integration/docker-compose.yml`` uses port 7688 so both can
run in parallel."""

PHASE1_AUTH: Final[tuple[str, str]] = ("neo4j", "benchmarkpassword")
"""Bolt credentials. Pinned across runs so contributors can connect
manually via ``cypher-shell`` for debugging."""

PHASE1_DATABASE: Final[str] = "neo4j"

_COMPOSE_PROJECT: Final[str] = "cygnet-benchmark"


def _bundled_compose_path() -> Path:
    """Resolve the bundled ``docker-compose.yml`` shipped with the
    package. Uses :mod:`importlib.resources` so it works from any
    installed location (editable, wheel, zipapp)."""
    return Path(str(files("cygnet.dev").joinpath("docker-compose.yml")))


def _ensure_docker_on_path() -> None:
    """Prepend Docker Desktop's bin to PATH when ``docker`` isn't
    resolvable. No-op on systems where ``docker`` is already on PATH
    (Linux native Docker, macOS Homebrew, etc.)."""
    if shutil.which("docker") is not None:
        return
    candidates = [
        Path("C:/Program Files/Docker/Docker/resources/bin"),
        Path("C:/Program Files (x86)/Docker/Docker/resources/bin"),
    ]
    for candidate in candidates:
        if (candidate / "docker.exe").exists():
            os.environ["PATH"] = f"{candidate}{os.pathsep}{os.environ.get('PATH', '')}"
            return


_ensure_docker_on_path()


class Phase1Neo4j:
    """Manage the dev-time Neo4j Docker container.

    Usage::

        with Phase1Neo4j() as neo:
            neo.clear()
            schema = neo.load_schema(spec_path)
            ...

    Or imperatively::

        neo = Phase1Neo4j()
        neo.start()
        try:
            ...
        finally:
            neo.stop()

    The class owns the container lifecycle, the driver connection,
    and the small set of Cypher-side mutation primitives
    (``clear`` for wipe between runs, ``load_schema`` for applying
    a schema spec's constraints and indexes, ``load_dataset`` as
    an extension point for caller-supplied loaders).

    Args:
        startup_timeout_s: maximum seconds to wait for ``RETURN 1``
            to succeed after ``docker compose up``. Default 120s,
            tuned for a cold Neo4j 5 + APOC start on a fast machine
            with some buffer for slower CI.
        compose_file: optional path to a docker-compose file. When
            ``None`` (default), uses the bundled file shipped as
            package data with :mod:`cygnet.dev`. Provide a custom
            path for project-specific Neo4j configurations.
    """

    bolt_uri: str = PHASE1_BOLT_URI
    auth: tuple[str, str] = PHASE1_AUTH
    database: str = PHASE1_DATABASE

    def __init__(
        self,
        *,
        startup_timeout_s: float = 120.0,
        compose_file: Path | None = None,
    ) -> None:
        self._startup_timeout_s = startup_timeout_s
        self._driver: Driver | None = None
        self._compose_file: Path = compose_file or _bundled_compose_path()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bring the container up and wait for ``RETURN 1`` to succeed.

        Idempotent: a second call against an already-running container
        no-ops the compose-up and re-validates responsiveness.
        """
        self._compose_up()
        self._wait_responsive()

    def stop(self, *, remove_volumes: bool = False) -> None:
        """Tear the container down. ``remove_volumes=True`` also drops
        the named volumes (next ``start`` then re-initialises an empty
        database). Default False to preserve loaded datasets across
        repeat runs."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
        args = [
            "docker",
            "compose",
            "-p",
            _COMPOSE_PROJECT,
            "-f",
            str(self._compose_file),
            "down",
        ]
        if remove_volumes:
            args.append("-v")
        subprocess.run(args, check=False)

    def is_running(self) -> bool:
        """Return True if the container responds to ``RETURN 1``."""
        try:
            driver = GraphDatabase.driver(self.bolt_uri, auth=self.auth)
        except Exception:
            return False
        try:
            with driver.session(database=self.database) as session:
                session.run("RETURN 1").single()
        except (ServiceUnavailable, Exception):
            return False
        finally:
            driver.close()
        return True

    def __enter__(self) -> Phase1Neo4j:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Driver access
    # ------------------------------------------------------------------

    @property
    def driver(self) -> Driver:
        """Cached driver. Constructed on first access; closed by
        :meth:`stop`."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.bolt_uri, auth=self.auth)
        return self._driver

    # ------------------------------------------------------------------
    # Mutation primitives
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete every node, relationship, constraint, and non-LOOKUP
        index. Faster than ``stop(remove_volumes=True)`` for wipes
        between runs; the container stays warm."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            for record in list(session.run("SHOW CONSTRAINTS")):
                session.run(f"DROP CONSTRAINT `{record['name']}`")
            for record in list(session.run("SHOW INDEXES WHERE type <> 'LOOKUP'")):
                session.run(f"DROP INDEX `{record['name']}`")

    def load_schema(self, spec_path: Path) -> Schema:
        """Apply a schema spec's constraints and indexes to the
        container, returning the loaded :class:`cygnet.Schema`.

        Callers can drop a schema spec in and get a ready-to-validate-
        against graph.
        """
        from cygnet.schema import load_schema_spec

        schema = load_schema_spec(spec_path)
        with self.driver.session(database=self.database) as session:
            for c in schema.constraints:
                if not c.property:
                    continue
                stmt = self._constraint_ddl(c)
                if stmt is None:
                    continue
                try:
                    session.run(stmt)
                except Exception:
                    # Community Edition rejects existence and node-key
                    # constraints; accept the permissive default same
                    # as the integration suite.
                    logger.debug("constraint %s skipped (CE rejection)", c.identifier)
            for i, ix in enumerate(schema.indexes):
                if ix.type.upper() not in {"BTREE", "RANGE"}:
                    continue
                prop_list = ", ".join(f"n.`{p}`" for p in ix.properties)
                session.run(
                    f"CREATE INDEX `_bench_idx_{i}` IF NOT EXISTS "
                    f"FOR (n:`{ix.label_or_rel}`) ON ({prop_list})"
                )
        return schema

    def load_dataset(
        self,
        loader: DatasetLoader,
    ) -> int:
        """Invoke a caller-supplied dataset loader and return the
        post-load node count.

        Callers pass their own loader::

            n = neo.load_dataset(my_loader)

        Args:
            loader: callable taking ``(driver, *, database="neo4j")``
                and returning the post-load node count. The driver
                is this instance's cached one; the database is
                :attr:`database` (``"neo4j"`` by default).
                ``database`` is passed as a keyword argument so
                loaders with keyword-only ``database`` work without
                a positional/keyword mismatch.

        Returns:
            The integer return value of ``loader``.
        """
        return loader(self.driver, database=self.database)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _constraint_ddl(c: object) -> str | None:
        from cygnet.models import SchemaConstraint

        if not isinstance(c, SchemaConstraint):
            return None
        if not c.property:
            return None
        type_upper = c.type.upper()
        if "UNIQUENESS" in type_upper or "UNIQUE" in type_upper:
            return (
                f"CREATE CONSTRAINT `{c.identifier}` IF NOT EXISTS "
                f"FOR (n:`{c.label_or_rel}`) REQUIRE n.`{c.property}` IS UNIQUE"
            )
        if "EXISTENCE" in type_upper or "NOT NULL" in type_upper:
            return (
                f"CREATE CONSTRAINT `{c.identifier}` IF NOT EXISTS "
                f"FOR (n:`{c.label_or_rel}`) REQUIRE n.`{c.property}` IS NOT NULL"
            )
        if "NODE_KEY" in type_upper or "NODE KEY" in type_upper:
            return (
                f"CREATE CONSTRAINT `{c.identifier}` IF NOT EXISTS "
                f"FOR (n:`{c.label_or_rel}`) REQUIRE n.`{c.property}` IS NODE KEY"
            )
        return None

    def _compose_up(self) -> None:
        subprocess.run(
            [
                "docker",
                "compose",
                "-p",
                _COMPOSE_PROJECT,
                "-f",
                str(self._compose_file),
                "up",
                "-d",
            ],
            check=True,
        )

    def _wait_responsive(self) -> None:
        """Poll ``RETURN 1`` until success or timeout."""
        deadline = time.monotonic() + self._startup_timeout_s
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                driver = GraphDatabase.driver(self.bolt_uri, auth=self.auth)
                with driver.session(database=self.database) as session:
                    session.run("RETURN 1").single()
                driver.close()
                logger.info("Phase1Neo4j ready at %s", self.bolt_uri)
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(1.5)
        raise TimeoutError(
            f"Phase1Neo4j at {self.bolt_uri} did not become responsive within "
            f"{self._startup_timeout_s}s. Last error: {last_exc!r}"
        )
