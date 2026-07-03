"""
loader.py
------------------------
Medical Report Loader

Loads PDF reports using LangChain's
PyPDFLoader.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:

    def __init__(self, pdf_path: str):

        
        self.pdf_path = Path(pdf_path)

    def load(self):

        if not self.pdf_path.exists():

            raise FileNotFoundError(
                f"\nPDF not found:\n{self.pdf_path}"
            )

        loader = PyPDFLoader(str(self.pdf_path))

        documents = loader.load()

        print("\n" + "=" * 60)
        print("Medical Report Loaded Successfully")
        print("=" * 60)

        print(f"File Name : {self.pdf_path.name}")

        print(f"Pages     : {len(documents)}")

        print("=" * 60 + "\n")

        return documents