"""
llm.py
---------------------------------------------------------
LLM Manager

Supports

1. Google Gemini
2. Ollama (Local Only)

Automatically detects whether Ollama is available.

Author : Himanshu Rajak
Project : AI Medico Bot
"""

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---------------------------------------------------------
# Detect Ollama Availability
# ---------------------------------------------------------

OLLAMA_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    pass


class LLMManager:

    # -----------------------------------------------------
    # Supported Models
    # -----------------------------------------------------

    GEMINI_MODELS = {
        "Gemini 2.5 Flash": "gemini-2.5-flash",
        "Gemini 2.5 Pro": "gemini-2.5-pro",
    }

    OLLAMA_MODELS = {
        "Llama 3.2": "llama3.2",
        "Mistral": "mistral",
        "Gemma 3": "gemma3",
        "Phi 4": "phi4",
    }

    # -----------------------------------------------------
    # Constructor
    # -----------------------------------------------------

    def __init__(
        self,
        provider="Google Gemini",
        model_name=None,
        temperature=0.2,
    ):

        self.provider = provider
        self.temperature = temperature

        if provider == "Google Gemini":

            if model_name is None:
                model_name = "Gemini 2.5 Flash"

            self.model_name = self.GEMINI_MODELS.get(
                model_name,
                "gemini-2.5-flash"
            )

        elif provider == "Ollama":

            if not OLLAMA_AVAILABLE:
                raise RuntimeError(
                    "Ollama is not installed or not supported in this environment."
                )

            if model_name is None:
                model_name = "Llama 3.2"

            self.model_name = self.OLLAMA_MODELS.get(
                model_name,
                "llama3.2"
            )

        else:
            raise ValueError(f"Unsupported Provider: {provider}")

    # -----------------------------------------------------
    # Load LLM
    # -----------------------------------------------------

    def get_llm(self):

        print("=" * 60)
        print("Loading LLM")
        print("=" * 60)

        print("Provider :", self.provider)
        print("Model    :", self.model_name)

        if self.provider == "Google Gemini":

            api_key = os.getenv("GOOGLE_API_KEY")

            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found. Check your .env or Streamlit Secrets."
                )

            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=api_key,
            )

        elif self.provider == "Ollama":

            llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
            )

        else:

            raise ValueError("Unsupported LLM Provider")

        print("LLM Loaded Successfully")
        print("=" * 60)

        return llm

    # -----------------------------------------------------
    # Available Providers
    # -----------------------------------------------------

    @staticmethod
    def available_providers():

        providers = ["Google Gemini"]

        if OLLAMA_AVAILABLE:
            providers.append("Ollama")

        return providers

    # -----------------------------------------------------
    # Available Models
    # -----------------------------------------------------

    @classmethod
    def available_models(cls, provider):

        if provider == "Google Gemini":
            return list(cls.GEMINI_MODELS.keys())

        elif provider == "Ollama":

            if OLLAMA_AVAILABLE:
                return list(cls.OLLAMA_MODELS.keys())

            return []

        return []

    # -----------------------------------------------------
    # Utility
    # -----------------------------------------------------

    @staticmethod
    def is_ollama_available():
        return OLLAMA_AVAILABLE