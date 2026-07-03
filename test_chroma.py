"""
Test ChromaDB
"""

from pathlib import Path

from utils.loader import PDFLoader
from utils.splitter import DocumentSplitter
from utils.embeddings import EmbeddingModel
from utils.vectorstore import VectorStore

pdf = list(Path("data").glob("*.pdf"))[0]

documents = PDFLoader(str(pdf)).load()

chunks = DocumentSplitter().split(documents)

embedding = EmbeddingModel(
    "BGE Small"
).get_embedding_model()

db = VectorStore(
    embedding
).create(
    chunks,
    db_type="ChromaDB"
)

results = db.similarity_search(
    "Hemoglobin",
    k=3
)

print("=" * 60)
print("Retrieved Chunks")
print("=" * 60)

for i, doc in enumerate(results, 1):

    print(f"\nChunk {i}\n")

    print(doc.page_content[:500])