"""
rag_chain.py
----------------------------------------
RAG Pipeline

Supports

1. Medical Q&A
2. Report Summary
3. Medical Term Explanation
4. Questions for Doctor

Author : Himanshu Rajak
Project : AI Medico Bot
"""

from utils.prompts import PromptManager
from utils.llm import LLMManager


class RAGChain:

    # --------------------------------------------------
    # Constructor
    # --------------------------------------------------

    def __init__(
        self,
        vector_db,
        provider="Google Gemini",
        model_name=None,
        temperature=0.2
    ):

        self.vector_db = vector_db

        self.llm = LLMManager(

            provider=provider,

            model_name=model_name,

            temperature=temperature

        ).get_llm()

    # --------------------------------------------------
    # Retrieve Context
    # --------------------------------------------------

    def retrieve_context(
        self,
        query,
        k=3
    ):

        docs = self.vector_db.similarity_search(

            query,

            k=k

        )

        context = "\n\n".join(

            doc.page_content

            for doc in docs

        )

        return context, docs

    # --------------------------------------------------
    # Ask Question
    # --------------------------------------------------

    def ask(
        self,
        question,
        k=3
    ):

        context, docs = self.retrieve_context(

            question,

            k

        )

        prompt = PromptManager.medical_qa_prompt()

        chain = prompt | self.llm

        response = chain.invoke(

            {

                "context": context,

                "question": question

            }

        )

        return response.content, docs

    # --------------------------------------------------
    # Report Summary
    # --------------------------------------------------

    def summarize(
        self
    ):

        docs = self.vector_db.similarity_search(

            "Summarize this medical report",

            k=10

        )

        context = "\n\n".join(

            doc.page_content

            for doc in docs

        )

        prompt = PromptManager.summary_prompt()

        chain = prompt | self.llm

        response = chain.invoke(

            {

                "context": context

            }

        )

        return response.content

    # --------------------------------------------------
    # Explain Medical Term
    # --------------------------------------------------

    def explain_term(
        self,
        term,
        k=5
    ):

        context, docs = self.retrieve_context(

            term,

            k

        )

        prompt = PromptManager.term_explanation_prompt()

        chain = prompt | self.llm

        response = chain.invoke(

            {

                "context": context,

                "question": term

            }

        )

        return response.content, docs

    # --------------------------------------------------
    # Questions for Doctor
    # --------------------------------------------------

    def doctor_questions(
        self
    ):

        docs = self.vector_db.similarity_search(

            "medical report",

            k=10

        )

        context = "\n\n".join(

            doc.page_content

            for doc in docs

        )

        prompt = PromptManager.doctor_questions_prompt()

        chain = prompt | self.llm

        response = chain.invoke(

            {

                "context": context

            }

        )

        return response.content

    # --------------------------------------------------
    # Disclaimer
    # --------------------------------------------------

    @staticmethod
    def disclaimer():

        return PromptManager.disclaimer()