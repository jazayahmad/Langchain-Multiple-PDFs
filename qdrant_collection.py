from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
qdrant_client = QdrantClient("localhost", port=6333)

qdrant_client.recreate_collection(
    collection_name="pulin_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
print("Collection Created!")