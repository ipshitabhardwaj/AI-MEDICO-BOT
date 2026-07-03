"""
Test Embedding Model
"""

from utils.embeddings import EmbeddingModel

embedding = EmbeddingModel(
    "BGE Small"
).get_embedding_model()

vector = embedding.embed_query(
    "What is Hemoglobin?"
)

print("=" * 60)
print("Embedding Length :", len(vector))
print("=" * 60)

print(vector[:10])