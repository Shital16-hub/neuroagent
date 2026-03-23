import asyncio
from app.services.neo4j_service import Neo4jService

async def test():
    neo4j = Neo4jService()
    await neo4j.connect()
    print('Neo4j AuraDB: Connected ✅')
    await neo4j.close()

asyncio.run(test())