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

import logging
from typing import Any, Optional

import neo4j
from neo4j import AsyncGraphDatabase, AsyncDriver

from app.config import get_settings
from app.models.paper import Paper

logger = logging.getLogger(__name__)


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

    async def connect(self) -> None:
        """
        Open the async Neo4j driver and verify connectivity.

        Called once at application startup via FastAPI lifespan.
        AuraDB URIs (neo4j+s://) handle TLS via the URI scheme.
        TrustAll() is required on Windows where Python's SSL certificate
        store does not include the AuraDB certificate chain, causing
        ssl.SSLCertVerificationError on self-signed certs.
        """
        settings = get_settings()
        logger.info("Connecting to Neo4j | uri_prefix=%s", settings.neo4j_uri[:30])

        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            trusted_certificates=neo4j.TrustAll(),
        )

        # Verify the connection is live
        await self._driver.verify_connectivity()
        logger.info("Neo4j connected | uri=%s", settings.neo4j_uri)

        await self._ensure_constraints()

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
        async with self.driver.session() as session:
            await session.run(
                query,
                paper_id=paper.paper_id,
                title=paper.title,
                year=paper.year,
                source=paper.source,
                doi=paper.doi,
                pdf_url=paper.pdf_url,
                authors=paper.authors,
            )
        logger.debug("Paper node upserted | paper_id=%s", paper.paper_id)

    # ── Concept Nodes ──────────────────────────────────────────────────────────

    async def save_concept_node(self, concept: str) -> None:
        """Upsert a Concept node by name."""
        query = "MERGE (:Concept {name: $name})"
        async with self.driver.session() as session:
            await session.run(query, name=concept.lower().strip())

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
        async with self.driver.session() as session:
            await session.run(query, paper_id=paper_id, concept=concept.lower().strip())

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
        async with self.driver.session() as session:
            await session.run(query, paper_id=paper_id, concept=concept.lower().strip())

    async def run_query(
        self, cypher: str, parameters: Optional[dict] = None
    ) -> list[dict]:
        """
        Run an arbitrary Cypher query and return results as a list of dicts.
        Useful for ad-hoc queries and the /api/graph/concepts endpoint.
        """
        async with self.driver.session() as session:
            result = await session.run(cypher, parameters or {})
            return await result.data()

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
        async with self.driver.session() as session:
            await session.run(
                query,
                concept_a=concept_a.lower().strip(),
                concept_b=concept_b.lower().strip(),
            )

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
        async with self.driver.session() as session:
            await session.run(query, citing_id=citing_id, cited_id=cited_id)

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
        async with self.driver.session() as session:
            result = await session.run(query, paper_ids=paper_ids)
            records = await result.data()
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
        async with self.driver.session() as session:
            result = await session.run(query, paper_id=paper_id)
            return await result.data()

    # ── Schema Constraints ─────────────────────────────────────────────────────

    async def _ensure_constraints(self) -> None:
        """
        Create uniqueness constraints on Paper.paper_id and Concept.name.
        Idempotent — safe to call on every startup.
        """
        constraints = [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
        ]
        async with self.driver.session() as session:
            for constraint in constraints:
                await session.run(constraint)
        logger.debug("Neo4j constraints verified")
