"""
llm.py
----------------------------------------
LLM Manager

Supports

1. Google Gemini
2. Ollama

Author : Himanshu Rajak
Project : AI Medico Bot
"""

import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


class LLMManager:

    # ----------------------------------
    # Supported Models
    # ----------------------------------

    GEMINI_MODELS = {

        "Gemini 2.5 Flash": "gemini-2.5-flash",

        "Gemini 2.5 Pro": "gemini-2.5-pro"

    }

    OLLAMA_MODELS = {

        "Llama 3.2": "llama3.2",

        "Mistral": "mistral",

        "Gemma 3": "gemma3",

        "Phi 4": "phi4"

    }

    # ----------------------------------
    # Constructor
    # ----------------------------------

    def __init__(

        self,

        provider="Google Gemini",

        model_name=None,

        temperature=0.2

    ):

        self.provider = provider

        self.temperature = temperature

        if provider == "Google Gemini":

            if model_name is None:

                model_name = "Gemini 2.5 Flash"

            self.model_name = self.GEMINI_MODELS[model_name]

        elif provider == "Ollama":

            if model_name is None:

                model_name = "Llama 3.2"

            self.model_name = self.OLLAMA_MODELS[model_name]

        else:

            raise ValueError("Unsupported LLM Provider")

    # ----------------------------------
    # Load LLM
    # ----------------------------------

    def get_llm(self):

        print("\n" + "=" * 60)
        print("Loading LLM")
        print("=" * 60)

        print(f"Provider : {self.provider}")

        print(f"Model    : {self.model_name}")

        if self.provider == "Google Gemini":

            llm = ChatGoogleGenerativeAI(

                model=self.model_name,

                temperature=self.temperature,

                google_api_key=os.getenv("GOOGLE_API_KEY")

            )

        elif self.provider == "Ollama":

            llm = ChatOllama(

                model=self.model_name,

                temperature=self.temperature

            )

        else:

            raise ValueError("Unsupported Provider")

        print("\nLLM Loaded Successfully")
        print("=" * 60 + "\n")

        return llm

    # ----------------------------------
    # Available Providers
    # ----------------------------------

    @staticmethod
    def available_providers():

        return [

            "Google Gemini",

            "Ollama"

        ]

    # ----------------------------------
    # Available Models
    # ----------------------------------

    @classmethod
    def available_models(cls, provider):

        if provider == "Google Gemini":

            return list(cls.GEMINI_MODELS.keys())

        elif provider == "Ollama":

            return list(cls.OLLAMA_MODELS.keys())

        else:

            return []