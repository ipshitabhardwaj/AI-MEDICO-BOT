"""
Test PDF Loader
"""

from pathlib import Path

from utils.loader import PDFLoader

pdf_files = list(Path("data").glob("*.pdf"))

if len(pdf_files) == 0:
    print("No PDF Found.")
    exit()

loader = PDFLoader(str(pdf_files[0]))

documents = loader.load()

print("=" * 60)
print("Total Pages :", len(documents))
print("=" * 60)

print(documents[0].page_content[:1000])