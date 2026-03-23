"""
Neo4j service — async graph database client.

Manages the knowledge graph: paper nodes, concept nodes, and the
relationships between them (cites, contains, related_to).

Supports both local bolt:// and AuraDB cloud neo4j+s:// connections.
The neo4j+s:// URI enables TLS automatically. TrustAll() is passed to
the driver to handle Windows SSL certificate store gaps with AuraDB.

Usage:
    from app.services.neo4j_service import Neo4jService

    service = Neo4jService()
    await service.connect()
    await service.save_paper_node(paper)
    await service.close()
"""

import asyncio
import logging
from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from app.config import get_settings
from app.models.paper import Paper

logger = logging.getLogger(__name__)

_RETRYABLE = (ServiceUnavailable, SessionExpired)


class Neo4jService:
    """
    Async Neo4j client using the official neo4j Python driver.

    URI is read exclusively from settings.neo4j_uri — never hardcoded.
    Works transparently for:
      - Local:   bolt://localhost:7687
      - AuraDB:  neo4j+s://xxxxxxxx.databases.neo4j.io
    """

    def __init__(self) -> None:
        self._driver: Optional[AsyncDriver] = None
        self._uri: str = ""
        self._settings = get_settings()

    async def connect(self) -> None:
        """
        Open the async Neo4j driver and verify connectivity.

        Called once at application startup via FastAPI lifespan.
        AuraDB URIs (neo4j+s://) handle TLS via the URI scheme.

        Connection pool settings prevent "defunct connection" errors on AuraDB:
          - max_connection_lifetime=1800: retire connections after 30 min so
            the pool never hands out a socket that AuraDB already closed (~60 min idle limit).
          - keep_alive=True: TCP keepalives stop NAT/firewall from silently
            dropping idle connections mid-session.
          - connection_acquisition_timeout=30: fail fast if the pool is exhausted.
        """
        settings = self._settings

        # On Windows, Python's SSL store lacks AuraDB's intermediate CA.
        # neo4j+ssc:// uses TLS encryption but skips chain verification.
        uri = settings.neo4j_uri.replace("neo4j+s://", "neo4j+ssc://", 1)
        uri = uri.replace("bolt+s://", "bolt+ssc://", 1)
        self._uri = uri

        logger.info("Connecting to Neo4j | uri_prefix=%s", uri[:30])
        await self._create_driver()
        await self._ensure_constraints()

    async def _create_driver(self) -> None:
        """Create (or recreate) the driver with pool settings and verify connectivity."""
        settings = self._settings
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_lifetime=1800,       # 30 min — retire before AuraDB's ~60 min idle cutoff
            max_connection_pool_size=10,
            connection_acquisition_timeout=30.0,
            connection_timeout=10.0,
            keep_alive=True,                    # TCP keepalives prevent silent NAT drops
        )
        await self._driver.verify_connectivity()
        logger.info("Neo4j connected | uri=%s", settings.neo4j_uri)

    async def _reconnect(self) -> None:
        """Close the stale driver and open a fresh one."""
        logger.warning("Neo4j: reconnecting after defunct connection")
        try:
            if self._driver:
                await self._driver.close()
        except Exception:
            pass
        await self._create_driver()

    async def _run(self, query: str, params: Optional[dict] = None) -> list[dict]:
        """
        Execute a Cypher query and return result rows as dicts.

        Retries once with a fresh connection on ServiceUnavailable / SessionExpired
        (the two exceptions the driver raises for defunct/dropped connections).
        """
        for attempt in range(2):
            try:
                async with self.driver.session() as session:
                    result = await session.run(query, params or {})
                    return await result.data()
            except _RETRYABLE as exc:
                if attempt == 1:
                    raise
                logger.warning(
                    "Neo4j defunct connection on attempt %d — reconnecting | error=%s",
                    attempt + 1,
                    exc,
                )
                await asyncio.sleep(0.5)
                await self._reconnect()
        return []  # unreachable, satisfies type checker

    async def close(self) -> None:
        """Close the driver. Called at application shutdown."""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j connection closed")

    @property
    def driver(self) -> AsyncDriver:
        """Return the driver, raising if not connected."""
        if self._driver is None:
            raise RuntimeError("Neo4jService.connect() has not been called")
        return self._driver

    # ── Paper Nodes ────────────────────────────────────────────────────────────

    async def save_paper_node(self, paper: Paper) -> None:
        """
        Upsert a Paper node in the graph.
        MERGE ensures no duplicate nodes are created on repeated calls.
        """
        query = """
        MERGE (p:Paper {paper_id: $paper_id})
        SET p.title     = $title,
            p.year      = $year,
            p.source    = $source,
            p.doi       = $doi,
            p.pdf_url   = $pdf_url,
            p.authors   = $authors
        """
        await self._run(query, dict(
            paper_id=paper.paper_id,
            title=paper.title,
            year=paper.year,
            source=paper.source,
            doi=paper.doi,
            pdf_url=paper.pdf_url,
            authors=paper.authors,
        ))
        logger.debug("Paper node upserted | paper_id=%s", paper.paper_id)

    # ── Concept Nodes ──────────────────────────────────────────────────────────

    async def save_concept_node(self, concept: str) -> None:
        """Upsert a Concept node by name."""
        await self._run("MERGE (:Concept {name: $name})", {"name": concept.lower().strip()})

    async def link_paper_to_concept(self, paper_id: str, concept: str) -> None:
        """
        Create a CONTAINS relationship: (Paper)-[:CONTAINS]->(Concept).
        Both nodes are created if they don't exist.
        """
        query = """
        MERGE (p:Paper {paper_id: $paper_id})
        MERGE (c:Concept {name: $concept})
        MERGE (p)-[:CONTAINS]->(c)
        """
        await self._run(query, {"paper_id": paper_id, "concept": concept.lower().strip()})

    async def link_paper_to_concept_mentions(self, paper_id: str, concept: str) -> None:
        """
        Create a MENTIONS relationship: (Paper)-[:MENTIONS]->(Concept).
        Used by ConceptExtractorAgent — semantically: a paper mentions a concept.
        """
        query = """
        MERGE (p:Paper {paper_id: $paper_id})
        MERGE (c:Concept {name: $concept})
        MERGE (p)-[:MENTIONS]->(c)
        """
        await self._run(query, {"paper_id": paper_id, "concept": concept.lower().strip()})

    async def run_query(
        self, cypher: str, parameters: Optional[dict] = None
    ) -> list[dict]:
        """
        Run an arbitrary Cypher query and return results as a list of dicts.
        Useful for ad-hoc queries and the /api/graph/concepts endpoint.
        """
        return await self._run(cypher, parameters)

    async def link_concepts(self, concept_a: str, concept_b: str) -> None:
        """
        Create a RELATED_TO relationship between two concepts (undirected).
        Uses MERGE so the relationship is created at most once.
        """
        query = """
        MERGE (a:Concept {name: $concept_a})
        MERGE (b:Concept {name: $concept_b})
        MERGE (a)-[:RELATED_TO]-(b)
        """
        await self._run(query, {
            "concept_a": concept_a.lower().strip(),
            "concept_b": concept_b.lower().strip(),
        })

    # ── Citation Edges ─────────────────────────────────────────────────────────

    async def link_citation(self, citing_id: str, cited_id: str) -> None:
        """
        Create a CITES relationship: (Paper)-[:CITES]->(Paper).
        Used to build the citation graph from Semantic Scholar data.
        """
        query = """
        MERGE (a:Paper {paper_id: $citing_id})
        MERGE (b:Paper {paper_id: $cited_id})
        MERGE (a)-[:CITES]->(b)
        """
        await self._run(query, {"citing_id": citing_id, "cited_id": cited_id})

    # ── Graph Queries ──────────────────────────────────────────────────────────

    async def get_concepts_for_session(self, paper_ids: list[str]) -> list[str]:
        """
        Return all unique concepts associated with a list of papers.
        Used by the API's /api/graph/concepts endpoint.
        """
        query = """
        MATCH (p:Paper)-[:CONTAINS]->(c:Concept)
        WHERE p.paper_id IN $paper_ids
        RETURN DISTINCT c.name AS concept
        ORDER BY concept
        """
        records = await self._run(query, {"paper_ids": paper_ids})
        return [r["concept"] for r in records]

    async def get_related_papers(self, paper_id: str, depth: int = 2) -> list[dict[str, Any]]:
        """
        Return papers connected by shared concepts within `depth` hops.
        Useful for discovering related work the query didn't surface directly.
        """
        query = """
        MATCH (p:Paper {paper_id: $paper_id})-[:CONTAINS*1..2]->(c:Concept)<-[:CONTAINS]-(related:Paper)
        WHERE related.paper_id <> $paper_id
        RETURN DISTINCT related.paper_id AS paper_id,
                        related.title    AS title,
                        related.year     AS year
        LIMIT 20
        """
        return await self._run(query, {"paper_id": paper_id})

    # ── Batch writes ───────────────────────────────────────────────────────────

    async def write_concept_graph(
        self,
        papers: list["Paper"],
        concepts: list[str],
        paper_ids: set[str],
    ) -> None:
        """
        Batch-upsert Paper nodes, Concept nodes, and MENTIONS edges in 3 queries.

        Replaces 200+ individual round-trips with 3 UNWIND statements, which
        dramatically reduces the chance of hitting a defunct connection mid-batch
        and cuts total latency by ~90%.
        """
        clean_concepts = [c.lower().strip() for c in concepts]

        # 1. Upsert all Paper nodes
        if papers:
            await self._run(
                """
                UNWIND $papers AS p
                MERGE (n:Paper {paper_id: p.paper_id})
                SET n.title   = p.title,
                    n.year    = p.year,
                    n.source  = p.source,
                    n.doi     = p.doi,
                    n.pdf_url = p.pdf_url,
                    n.authors = p.authors
                """,
                {
                    "papers": [
                        {
                            "paper_id": p.paper_id,
                            "title": p.title,
                            "year": p.year,
                            "source": p.source,
                            "doi": p.doi,
                            "pdf_url": p.pdf_url,
                            "authors": p.authors,
                        }
                        for p in papers
                    ]
                },
            )

        # 2. Upsert all Concept nodes
        if clean_concepts:
            await self._run(
                "UNWIND $names AS name MERGE (:Concept {name: name})",
                {"names": clean_concepts},
            )

        # 3. MENTIONS edges — each paper that was summarised × each concept
        if paper_ids and clean_concepts:
            edges = [
                {"paper_id": pid, "concept": c}
                for pid in paper_ids
                for c in clean_concepts
            ]
            await self._run(
                """
                UNWIND $edges AS e
                MATCH (p:Paper {paper_id: e.paper_id})
                MATCH (c:Concept {name: e.concept})
                MERGE (p)-[:MENTIONS]->(c)
                """,
                {"edges": edges},
            )

        logger.debug(
            "Neo4j batch write complete | papers=%d concepts=%d edges=%d",
            len(papers),
            len(clean_concepts),
            len(paper_ids) * len(clean_concepts),
        )

    # ── Schema Constraints ─────────────────────────────────────────────────────

    async def _ensure_constraints(self) -> None:
        """
        Create uniqueness constraints on Paper.paper_id and Concept.name.
        Idempotent — safe to call on every startup.
        """
        for constraint in [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
        ]:
            await self._run(constraint)
        logger.debug("Neo4j constraints verified")
