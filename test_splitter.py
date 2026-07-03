"""
Test Document Splitter
"""

from pathlib import Path

from utils.loader import PDFLoader
from utils.splitter import DocumentSplitter

pdf = list(Path("data").glob("*.pdf"))[0]

documents = PDFLoader(str(pdf)).load()

splitter = DocumentSplitter()

chunks = splitter.split(documents)

print("=" * 60)
print("Total Chunks :", len(chunks))
print("=" * 60)

print(chunks[0].page_content)