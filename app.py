"""
=========================================================
AI Medico Bot
---------------------------------------------------------
Intelligent Medical Report Assistant

Features
---------
• Upload Medical Reports
• Google Gemini & Ollama Support
• FAISS & ChromaDB
• Multiple Embedding Models
• Medical Report Summary
• Medical Q&A
• Explain Medical Terms
• Questions for Doctor
• Retrieved Context Viewer

Author : Ipshita Bhardwaj
=========================================================
"""

# =========================================================
# Imports
# =========================================================

import os
import time
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils import (
    PDFLoader,
    DocumentSplitter,
    EmbeddingModel,
    VectorStore,
    LLMManager,
    PromptManager,
    RAGChain,
)

# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="AI Medico Bot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Visual identity — "The Chart"
# A clinical-report aesthetic: sections read like panels
# of a diagnostic chart, values read like lab readouts.
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --paper: #F5F4EF;
    --panel: #FFFFFF;
    --ink: #1E2730;
    --ink-soft: #5B6570;
    --clinical: #0F6E5C;
    --clinical-deep: #0A4B3E;
    --amber: #B4802E;
    --note: #A0443A;
    --line: #DDD9CC;
}

.stApp { background: var(--paper); color: var(--ink); font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Source Serif 4', serif !important; color: var(--clinical-deep) !important; }

/* header */
.chart-header {
    padding: 0.3rem 0 1rem 0;
    border-bottom: 2px solid var(--clinical-deep);
    margin-bottom: 1.4rem;
}
.chart-header .chart-title {
    font-family: 'Source Serif 4', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: var(--clinical-deep);
}
.chart-header .chart-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-top: 0.2rem;
}

/* panel label — used instead of st.subheader for section identity */
.panel-tab {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--clinical-deep);
    background: #E4EFEA;
    border: 1px solid var(--clinical);
    border-radius: 2px;
    padding: 0.15rem 0.55rem;
    margin-bottom: 0.5rem;
}
.panel-caption { color: var(--ink-soft); font-size: 0.9rem; margin-bottom: 0.8rem; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--clinical-deep); }
[data-testid="stSidebar"] * { color: #F2F0E6 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(242,240,230,0.15); }
[data-testid="stSidebar"] .stFileUploader section { background: #F2F0E6; border-radius: 3px; }
[data-testid="stSidebar"] .stFileUploader section * { color: var(--ink) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #F2F0E6 !important; color: var(--ink) !important; border-radius: 3px; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--ink) !important; }

/* buttons */
.stButton > button {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-radius: 3px;
    border: 1px solid var(--clinical);
    background: var(--panel);
    color: var(--clinical-deep);
    padding: 0.55rem 1rem;
}
.stButton > button:hover { background: var(--clinical-deep); color: #F2F0E6; border-color: var(--clinical-deep); }
[data-testid="stSidebar"] .stButton:nth-of-type(1) button {
    background: #D9C79E; color: var(--clinical-deep); border-color: #D9C79E; font-weight: 600;
}
[data-testid="stSidebar"] .stButton:nth-of-type(3) button {
    background: transparent; border: 1px dashed rgba(242,240,230,0.4);
}

/* metrics as lab readouts */
[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--line); border-radius: 3px;
    padding: 0.65rem 0.8rem 0.5rem 0.8rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.66rem !important;
    letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-soft) !important;
}
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: var(--clinical-deep) !important; }

/* info panels -> chart notes, not default blue boxes */
[data-testid="stAlertContentInfo"] {
    background: var(--panel) !important; border: 1px solid var(--line) !important;
    border-left: 3px solid var(--clinical) !important; border-radius: 2px !important; color: var(--ink) !important;
}

/* the medical disclaimer -> a chart annotation, not a shouting warning box */
[data-testid="stAlertContentWarning"] {
    background: var(--panel) !important; border: 1px solid var(--line) !important;
    border-left: 4px solid var(--note) !important; border-radius: 2px !important; color: var(--ink) !important;
}

/* expander = exhibit card */
[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 3px !important; background: var(--panel) !important; }
[data-testid="stExpander"] summary { font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; color: var(--clinical-deep) !important; }

/* source page refs */
.ref-row { margin-top: 0.5rem; }
.ref-chip {
    display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: var(--clinical-deep); border: 1.3px solid var(--clinical); border-radius: 3px;
    padding: 0.15rem 0.5rem; margin: 0 0.3rem 0.3rem 0;
}

/* colophon */
.colophon { text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: var(--ink-soft); padding: 1rem 0 0.4rem 0; letter-spacing: 0.03em; }
</style>
""",
    unsafe_allow_html=True
)

# =========================================================
# Header
# =========================================================

st.markdown(
    """
<div class="chart-header">
    <div class="chart-title">AI Medico Bot</div>
    <div class="chart-sub">Medical Report Assistant · LangChain · Gemini · Ollama · FAISS / ChromaDB</div>
</div>
""",
    unsafe_allow_html=True
)

# =========================================================
# Sidebar
# =========================================================

st.sidebar.markdown('<div class="panel-tab">Configuration</div>', unsafe_allow_html=True)

uploaded_pdf = st.sidebar.file_uploader(
    "Upload Medical Report",
    type=["pdf"]
)

provider = st.sidebar.selectbox(
    "LLM Provider",
    LLMManager.available_providers()
)

model_name = st.sidebar.selectbox(
    "Model",
    LLMManager.available_models(provider)
)

vector_db = st.sidebar.selectbox(
    "Vector Database",
    [
        "FAISS",
        "ChromaDB"
    ]
)

embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    EmbeddingModel.available_models()
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=500,
    max_value=2000,
    value=1000,
    step=100
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=500,
    value=200,
    step=50
)

top_k = st.sidebar.slider(
    "Top K Retrieval",
    min_value=1,
    max_value=10,
    value=3
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

build_database = st.sidebar.button(
    "Build Database",
    use_container_width=True
)

clear_chat = st.sidebar.button(
    "Clear Chat",
    use_container_width=True
)

delete_database = st.sidebar.button(
    "Delete Database",
    use_container_width=True
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
**AI Medico Bot** · v1.0

Supports Google Gemini, Ollama, FAISS, ChromaDB, and multiple embedding models.

Educational use only.
"""
)


# =========================================================
# Session State Initialization
# =========================================================

if "rag" not in st.session_state:
    st.session_state.rag = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "database_ready" not in st.session_state:
    st.session_state.database_ready = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

if "vector_db" not in st.session_state:
    st.session_state.vector_db = ""

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = ""

if "provider" not in st.session_state:
    st.session_state.provider = ""

if "llm_model" not in st.session_state:
    st.session_state.llm_model = ""

if "report_summary" not in st.session_state:
    st.session_state.report_summary = ""

if "doctor_questions" not in st.session_state:
    st.session_state.doctor_questions = ""


# =========================================================
# Build Vector Database
# =========================================================

if build_database:

    if uploaded_pdf is None:

        st.warning("Please upload a medical report first.")

    else:

        data_folder = Path("data")
        data_folder.mkdir(exist_ok=True)

        for file in data_folder.glob("*.pdf"):
            file.unlink()

        pdf_path = data_folder / uploaded_pdf.name

        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        progress = st.progress(0)
        status = st.empty()

        try:

            status.info("Loading medical report...")
            loader = PDFLoader(str(pdf_path))
            documents = loader.load()
            progress.progress(15)

            status.info("Splitting report into chunks...")
            splitter = DocumentSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split(documents)
            progress.progress(35)

            status.info("Loading embedding model...")
            embedding = EmbeddingModel(embedding_model).get_embedding_model()
            progress.progress(55)

            status.info(f"Creating {vector_db} database...")
            vectorstore = VectorStore(embedding)
            db = vectorstore.create(chunks, db_type=vector_db)
            progress.progress(75)

            status.info("Initializing assistant...")
            rag = RAGChain(
                vector_db=db,
                provider=provider,
                model_name=model_name
            )
            progress.progress(100)

            st.session_state.rag = rag
            st.session_state.documents = documents
            st.session_state.chunks = chunks
            st.session_state.database_ready = True
            st.session_state.pdf_name = uploaded_pdf.name
            st.session_state.vector_db = vector_db
            st.session_state.embedding_model = embedding_model
            st.session_state.provider = provider
            st.session_state.llm_model = model_name

            status.success("Ready.")
            st.success(f"Processed {uploaded_pdf.name}")

            progress.empty()

        except Exception as e:

            st.error(f"Error : {e}")
            progress.empty()

# =========================================================
# Document Statistics
# =========================================================

if st.session_state.database_ready:

    st.markdown('<div class="panel-tab">Chart Summary</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Pages", len(st.session_state.documents))

    with col2:
        st.metric("Chunks", len(st.session_state.chunks))

    with col3:
        st.metric("Vector DB", st.session_state.vector_db)

    with col4:
        st.metric("LLM", st.session_state.provider)

    col5, col6 = st.columns(2)

    with col5:
        st.info(
f"""
**Report File**

{st.session_state.pdf_name}
"""
        )

    with col6:
        st.info(
f"""
**Configuration**

Embedding — {st.session_state.embedding_model}

LLM — {st.session_state.llm_model}
"""
        )


# =========================================================
# Medical Report Summary
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="panel-tab">Findings</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-caption">Generate a plain-language summary of the uploaded report.</div>', unsafe_allow_html=True)

generate_summary = st.button(
    "Generate Summary",
    use_container_width=True
)

if generate_summary:

    if not st.session_state.database_ready:
        st.warning("Please build the database first.")

    else:

        with st.spinner("Generating summary..."):

            try:
                summary = st.session_state.rag.summarize()
                st.session_state.report_summary = summary

            except Exception as e:
                st.error(e)

if st.session_state.report_summary:

    st.success("Summary generated")
    st.write(st.session_state.report_summary)


# =========================================================
# Medical Term Explanation
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="panel-tab">Term Lookup</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-caption">Enter a medical term to understand its meaning in simple language.</div>', unsafe_allow_html=True)

medical_term = st.text_input(
    "Medical Term",
    placeholder="Example: Hemoglobin, RBC, Creatinine...",
    label_visibility="collapsed"
)

explain_term = st.button(
    "Explain Term",
    use_container_width=True
)

if explain_term:

    if not st.session_state.database_ready:
        st.warning("Please build the database first.")

    elif medical_term.strip() == "":
        st.warning("Please enter a medical term.")

    else:

        with st.spinner("Searching report..."):

            try:
                answer, docs = st.session_state.rag.explain_term(
                    medical_term,
                    k=top_k
                )

                st.session_state.retrieved_docs = docs

                st.success("Explanation generated")
                st.write(answer)

            except Exception as e:
                st.error(e)

# =========================================================
# Medical Q&A
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="panel-tab">Consultation Log</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-caption">Ask any question related to the uploaded report.</div>', unsafe_allow_html=True)

question = st.text_input(
    "Ask your question",
    placeholder="Example: What is Hemoglobin? Explain my cholesterol level.",
    label_visibility="collapsed"
)

ask_question = st.button(
    "Ask",
    use_container_width=True
)

if ask_question:

    if not st.session_state.database_ready:
        st.warning("Please build the database first.")

    elif question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Analyzing report..."):

            try:
                start = time.time()

                answer, docs = st.session_state.rag.ask(
                    question,
                    k=top_k
                )

                end = time.time()
                response_time = end - start

                st.session_state.retrieved_docs = docs

                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "assistant", "content": answer})

                st.success("Answer")
                st.write(answer)
                st.caption(f"Response time — {response_time:.2f}s")

            except Exception as e:
                st.error(e)

if len(st.session_state.messages) > 0:

    st.markdown('<div class="panel-tab">Conversation</div>', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# =========================================================
# Retrieved Chunks & Source Pages
# =========================================================

if len(st.session_state.retrieved_docs) > 0:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-tab">Retrieved Context</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-caption">Chunks retrieved from the vector database and provided to the model.</div>', unsafe_allow_html=True)

    for index, doc in enumerate(st.session_state.retrieved_docs, start=1):

        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Unknown")

        with st.expander(f"Exhibit {index:02d} — Page {page}"):

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Page", page)

            with col2:
                st.metric("Characters", len(doc.page_content))

            st.markdown("**Retrieved Text**")
            st.write(doc.page_content)

            st.markdown("---")
            st.markdown("**Metadata**")
            st.json(doc.metadata)

    st.markdown('<div class="panel-tab">Source Pages</div>', unsafe_allow_html=True)

    pages = sorted(
        {str(doc.metadata.get("page", "Unknown")) for doc in st.session_state.retrieved_docs}
    )

    chips = "".join(f'<span class="ref-chip">p. {p}</span>' for p in pages)

    st.markdown(f'<div class="ref-row">{chips}</div>', unsafe_allow_html=True)

# =========================================================
# Medical Disclaimer
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.warning(
    """
**Medical Disclaimer**

AI Medico Bot is intended for educational and informational purposes only.

- It does not diagnose diseases.
- It does not prescribe medications.
- It does not replace professional medical advice.
- Always consult a qualified healthcare professional for diagnosis and treatment.

Generated responses are based only on the uploaded report and the selected model.
"""
)

# =========================================================
# Footer
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
**Technology Stack**

Streamlit · LangChain · Python · HuggingFace
"""
    )

with col2:
    st.markdown(
        f"""
**AI Components**

Provider — {st.session_state.provider}

Model — {st.session_state.llm_model}

Embedding — {st.session_state.embedding_model}
"""
    )

with col3:
    st.markdown(
        f"""
**Vector Database**

Database — {st.session_state.vector_db}

Top K — {top_k}
"""
    )

st.markdown(
    """
<div class="colophon">
    AI Medico Bot &nbsp;·&nbsp; Retrieval-Augmented Generation &nbsp;·&nbsp;
    Gemini &nbsp;·&nbsp; Ollama &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; ChromaDB<br>
    Developed by Ipshita Bhardwaj &nbsp;·&nbsp; v1.0
</div>
""",
    unsafe_allow_html=True
)