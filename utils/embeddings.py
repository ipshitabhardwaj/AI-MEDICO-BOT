"""
embeddings.py
----------------------------------------
Embedding Model Manager

Supports multiple HuggingFace
Sentence Transformer models.

Author : Himanshu Rajak
Project : AI Medico Bot
"""

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    # ----------------------------------------------------
    # Available Embedding Models
    # ----------------------------------------------------

    MODELS = {

        "MiniLM":
        "sentence-transformers/all-MiniLM-L6-v2",

        "BGE Small":
        "BAAI/bge-small-en-v1.5",

        "BGE Base":
        "BAAI/bge-base-en-v1.5",

        "E5 Base":
        "intfloat/e5-base-v2",

        "MPNet":
        "sentence-transformers/all-mpnet-base-v2"

    }

    # ----------------------------------------------------
    # Constructor
    # ----------------------------------------------------

    def __init__(self, model_name="MiniLM"):

        if model_name not in self.MODELS:

            raise ValueError(
                f"Unsupported Embedding Model : {model_name}"
            )

        self.model_name = model_name

        self.model_path = self.MODELS[model_name]

    # ----------------------------------------------------
    # Load Embedding Model
    # ----------------------------------------------------

    def get_embedding_model(self):

        print("\n" + "=" * 60)
        print("Loading Embedding Model")
        print("=" * 60)

        print(f"Selected Model : {self.model_name}")

        print(f"Model Path     : {self.model_path}")

        embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_path
        )

        print("\nEmbedding Model Loaded Successfully")

        print("=" * 60 + "\n")

        return embedding_model

    # ----------------------------------------------------
    # Show Available Models
    # ----------------------------------------------------

    @classmethod
    def available_models(cls):

        return list(cls.MODELS.keys())