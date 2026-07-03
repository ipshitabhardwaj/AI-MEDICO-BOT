"""
Test Complete RAG Pipeline
"""

from pathlib import Path

from utils.loader import PDFLoader
from utils.splitter import DocumentSplitter
from utils.embeddings import EmbeddingModel
from utils.vectorstore import VectorStore
from utils.rag_chain import RAGChain

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
    db_type="FAISS"
)

rag = RAGChain(
    vector_db=db,
    provider="Google Gemini",
    model_name="Gemini 2.5 Flash"
)

answer, docs = rag.ask(
    "Summarize this medical report."
)

print("=" * 60)
print("RAG Response")
print("=" * 60)

print(answer)

print("\n")

print("=" * 60)
print("Retrieved Documents")
print("=" * 60)

for i, doc in enumerate(docs, 1):

    print(f"\nChunk {i}")

    print(doc.page_content[:400])