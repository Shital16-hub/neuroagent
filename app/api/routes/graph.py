"""
Graph API routes — knowledge graph exploration.

GET /api/graph/concepts  — Return concept nodes and edges for visualization.
GET /api/graph/papers    — Return papers related to a paper via shared concepts.
"""

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query, HTTPException, status

from app.dependencies import get_neo4j
from app.services.neo4j_service import Neo4jService

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


@router.get(
    "/concepts",
    summary="Concept graph",
    description=(
        "Returns concept nodes and RELATED_TO edges from the Neo4j knowledge graph. "
        "Designed for force-directed graph visualizations (D3.js, Gradio, etc.)."
    ),
)
async def get_concept_graph(
    limit: int = Query(default=50, ge=1, le=200, description="Max concepts to return"),
    neo4j: Optional[Neo4jService] = Depends(get_neo4j),
) -> dict[str, Any]:
    """Return concept nodes and RELATED_TO edges for graph visualization."""
    if neo4j is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j service is not available",
        )

    # Fetch concept nodes
    concepts_query = f"""
    MATCH (c:Concept)
    RETURN c.name AS name
    ORDER BY c.name
    LIMIT {limit}
    """
    concept_records = await neo4j.run_query(concepts_query)
    nodes = [{"id": r["name"], "label": r["name"], "type": "concept"} for r in concept_records]
    concept_names = {r["name"] for r in concept_records}

    # Fetch edges between returned concepts
    if concept_names:
        name_list = ", ".join(f'"{n}"' for n in concept_names)
        edges_query = f"""
        MATCH (a:Concept)-[:RELATED_TO]-(b:Concept)
        WHERE a.name IN [{name_list}] AND b.name IN [{name_list}]
        RETURN DISTINCT a.name AS source, b.name AS target
        """
        edge_records = await neo4j.run_query(edges_query)
        # Deduplicate undirected edges
        seen = set()
        edges = []
        for r in edge_records:
            key = tuple(sorted([r["source"], r["target"]]))
            if key not in seen:
                seen.add(key)
                edges.append({"source": r["source"], "target": r["target"]})
    else:
        edges = []

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


@router.get(
    "/papers/{paper_id}/related",
    summary="Related papers",
    description="Return papers related to the given paper via shared concepts in the knowledge graph.",
)
async def get_related_papers(
    paper_id: str,
    depth: int = Query(default=2, ge=1, le=3, description="Graph traversal depth"),
    neo4j: Optional[Neo4jService] = Depends(get_neo4j),
) -> dict[str, Any]:
    """Find papers related via shared concepts."""
    if neo4j is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j service is not available",
        )

    related = await neo4j.get_related_papers(paper_id=paper_id, depth=depth)
    return {
        "paper_id": paper_id,
        "related_papers": related,
        "count": len(related),
    }
