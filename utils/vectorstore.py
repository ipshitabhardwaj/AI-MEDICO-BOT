"""
vectorstore.py
----------------------------------------
Vector Database Manager

Supports

1. FAISS
2. ChromaDB

Author : Himanshu Rajak
Project : AI Medico Bot
"""

import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma


class VectorStore:

    # ------------------------------------
    # Constructor
    # ------------------------------------

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model

        self.faiss_path = "database/faiss_db"

        self.chroma_path = "database/chroma_db"

    # ------------------------------------
    # Create FAISS
    # ------------------------------------

    def create_faiss(self, chunks):

        print("\nCreating FAISS Vector Database...")

        db = FAISS.from_documents(
            chunks,
            self.embedding_model
        )

        db.save_local(self.faiss_path)

        print("FAISS Database Created Successfully")

        return db

    # ------------------------------------
    # Load FAISS
    # ------------------------------------

    def load_faiss(self):

        print("Loading FAISS Database...")

        return FAISS.load_local(
            self.faiss_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    # ------------------------------------
    # Create ChromaDB
    # ------------------------------------

    def create_chroma(self, chunks):

        print("\nCreating Chroma Database...")

        db = Chroma.from_documents(

            documents=chunks,

            embedding=self.embedding_model,

            persist_directory=self.chroma_path

        )

        print("Chroma Database Created Successfully")

        return db

    # ------------------------------------
    # Load ChromaDB
    # ------------------------------------

    def load_chroma(self):

        print("Loading Chroma Database...")

        return Chroma(

            persist_directory=self.chroma_path,

            embedding_function=self.embedding_model

        )

    # ------------------------------------
    # Create Database
    # ------------------------------------

    def create(self, chunks, db_type="FAISS"):

        db_type = db_type.lower()

        if db_type == "faiss":

            return self.create_faiss(chunks)

        elif db_type == "chromadb":

            return self.create_chroma(chunks)

        elif db_type == "chroma":

            return self.create_chroma(chunks)

        else:

            raise ValueError(
                "Unsupported Vector Database"
            )

    # ------------------------------------
    # Load Database
    # ------------------------------------

    def load(self, db_type="FAISS"):

        db_type = db_type.lower()

        if db_type == "faiss":

            return self.load_faiss()

        elif db_type == "chromadb":

            return self.load_chroma()

        elif db_type == "chroma":

            return self.load_chroma()

        else:

            raise ValueError(
                "Unsupported Vector Database"
            )

    # ------------------------------------
    # Delete Database
    # ------------------------------------

    def delete(self, db_type="FAISS"):

        db_type = db_type.lower()

        if db_type == "faiss":

            shutil.rmtree(
                self.faiss_path,
                ignore_errors=True
            )

            print("FAISS Database Deleted")

        elif db_type in ["chroma", "chromadb"]:

            shutil.rmtree(
                self.chroma_path,
                ignore_errors=True
            )

            print("Chroma Database Deleted")

        else:

            raise ValueError(
                "Unsupported Vector Database"
            )

    # ------------------------------------
    # Check Database Exists
    # ------------------------------------

    def exists(self, db_type="FAISS"):

        db_type = db_type.lower()

        if db_type == "faiss":

            return Path(self.faiss_path).exists()

        elif db_type in ["chroma", "chromadb"]:

            return Path(self.chroma_path).exists()

        return False