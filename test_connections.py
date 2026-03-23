import asyncio
from app.services.qdrant_service import QdrantService

async def clear():
      svc = QdrantService()
      await svc.connect()
      from app.config import get_settings
      s = get_settings()
      await svc._client.delete_collection(s.qdrant_papers_collection)
      print('Collection deleted — will be recreated on next query')
      await svc.close()

asyncio.run(clear())