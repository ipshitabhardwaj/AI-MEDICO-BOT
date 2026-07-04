https://ai-medico-bot-rag.streamlit.app/

# AI Medico Bot

A Retrieval-Augmented Generation assistant for reading and understanding medical reports. Upload a report, and get plain-language summaries, term explanations, and grounded answers to questions — with every response traceable back to the retrieved report text.

**For educational and informational use only. Not a substitute for professional medical advice.**

## Features

- Medical report (PDF) upload and ingestion
- Choice of LLM provider: Google Gemini or Ollama (local)
- Choice of vector store: FAISS or ChromaDB
- Multiple embedding model options
- Configurable chunking (chunk size, overlap) and Top-K retrieval
- Plain-language report summary generation
- Medical term lookup and explanation (e.g. "What is Hemoglobin?")
- Conversational Q&A over the report, with chat history
- Retrieved context viewer: source chunks, page numbers, and metadata for every answer
- Built-in medical disclaimer shown throughout the app

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| LLM | Google Gemini / Ollama |
| Embeddings | HuggingFace (multiple models) |
| Vector Store | FAISS / ChromaDB |

## Project Structure

```
AI_Medico_Bot/
├── app.py                   # Streamlit entry point
├── data/                    # Uploaded reports (created at runtime)
├── database/                # Persisted vector store
├── utils/
│   ├── loader.py             # PDFLoader
│   ├── splitter.py           # DocumentSplitter
│   ├── embeddings.py         # EmbeddingModel
│   ├── vectorstore.py        # VectorStore
│   ├── llm_manager.py        # LLMManager (provider/model selection)
│   ├── prompt_manager.py     # PromptManager
│   └── rag_chain.py          # RAGChain (ask / summarize / explain_term)
├── requirements.txt
└── .env                      # API keys (not committed)
```

## Setup

1. **Enter the project**

   ```bash
   cd AI_Medico_Bot
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv medico_env
   medico_env\Scripts\activate      # Windows
   source medico_env/bin/activate   # macOS / Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**

   Create a `.env` file in the project root:

   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

   If using Ollama, make sure the Ollama service is running locally and the required model is pulled (e.g. `ollama pull llama3`).

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

   The app will be available at `http://localhost:8501`.

## Usage

1. Upload a medical report (PDF) from the sidebar.
2. Select an LLM provider and model, a vector database, an embedding model, and adjust chunking / retrieval settings.
3. Click **Build Database**.
4. Use any of:
   - **Generate Summary** — a plain-language overview of the report
   - **Explain Term** — look up a specific medical term found in the report
   - **Ask** — free-form Q&A over the report
5. Expand **Retrieved Context** to see exactly which chunks, pages, and metadata informed each answer.
6. Use **Clear Chat** to reset the conversation, or **Delete Database** to remove the current index.

## Disclaimer

AI Medico Bot does not diagnose conditions, prescribe treatment, or replace a qualified healthcare professional. All responses are generated only from the uploaded report and the selected model, and should be independently verified with a doctor.

## Author

Ipshita Bhardwaj