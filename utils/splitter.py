"""
splitter.py
--------------------------------
Medical Report Text Splitter

Splits PDF documents into
smaller overlapping chunks
using LangChain's
RecursiveCharacterTextSplitter.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:

    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split(self, documents):

        chunks = self.text_splitter.split_documents(documents)

        print("\n" + "=" * 60)
        print("Document Split Successfully")
        print("=" * 60)

        print(f"Chunk Size      : {self.chunk_size}")
        print(f"Chunk Overlap   : {self.chunk_overlap}")
        print(f"Total Chunks    : {len(chunks)}")

        if len(chunks) > 0:

            avg_length = sum(
                len(chunk.page_content)
                for chunk in chunks
            ) / len(chunks)

            print(f"Average Length  : {avg_length:.2f} characters")

        print("=" * 60 + "\n")

        return chunks