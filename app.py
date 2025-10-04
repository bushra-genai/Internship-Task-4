# app_rag_streamlit.py
"""
Streamlit RAG App (polished)
- Upload PDF/TXT -> Chroma vectorstore (HuggingFace embeddings)
- Chat UI with history + download as PDF
- LLM backend: Groq (LLaMA model) or fallback mode
"""

import os
import io
import tempfile
from dotenv import load_dotenv
import streamlit as st
from typing import List
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# LangChain & loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Groq LLM
try:
    from langchain_groq import ChatGroq
    _HAS_GROQ = True
except Exception:
    _HAS_GROQ = False

# -------------------------
# Load environment
# -------------------------
load_dotenv()
DEFAULT_CHROMA_DIR = "./chroma_db"

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="RAG Q/A (Streamlit)", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    '<h1 style="color:red; font-weight:bold;">📚 RAG Q/A — Retrieval Augmented Generation </h1>',
    unsafe_allow_html=True
)

# -------------------------
# Sidebar: configuration
# -------------------------
st.sidebar.header("⚙️ Settings")
use_persist = st.sidebar.checkbox("Persist Chroma DB (./chroma_db)", value=True)
chroma_dir = st.sidebar.text_input(
    "Chroma persist directory",
    value=DEFAULT_CHROMA_DIR if use_persist else tempfile.mkdtemp()
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 API Key")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Retrieval settings")
embed_model_name = st.sidebar.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
top_k = st.sidebar.number_input("Top-k retrieved chunks", value=3, min_value=1, max_value=10, step=1)
chunk_size = st.sidebar.number_input("Chunk size", value=500, min_value=100, max_value=2000, step=50)
chunk_overlap = st.sidebar.number_input("Chunk overlap", value=50, min_value=0, max_value=500, step=10)

if st.sidebar.button("🗑️ Clear chat history"):
    st.session_state.clear()
    st.sidebar.success("Chat history cleared.")

st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 **Developed by Bushra Mubeen**")

# -------------------------
# Session state init
# -------------------------
if "vectorstore_exists" not in st.session_state:
    st.session_state.vectorstore_exists = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# -------------------------
# Upload section
# -------------------------
st.header("1) Upload Documents (PDF / TXT)")
uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt"], accept_multiple_files=True)

# Helper: load files
def load_file_to_docs(file_obj) -> List:
    name = file_obj.name
    suffix = os.path.splitext(name)[1].lower()
    try:
        if suffix == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_obj.getvalue())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp:
                tmp.write(file_obj.getvalue())
                tmp_path = tmp.name
            loader = TextLoader(tmp_path, encoding="utf-8")
        return loader.load()
    except Exception as e:
        st.error(f"❌ Failed to load {name}: {e}")
        return []

# Build vectorstore
if uploaded_files:
    all_docs = []
    for f in uploaded_files:
        docs = load_file_to_docs(f)
        if docs:
            all_docs.extend(docs)
            if f.name not in st.session_state.uploaded_docs:
                st.session_state.uploaded_docs.append(f.name)

    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splitted = text_splitter.split_documents(all_docs)

        with st.spinner("🔄 Creating embeddings & Chroma DB..."):
            embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
            vectorstore = Chroma.from_documents(
                documents=splitted,
                embedding=embeddings,
                persist_directory=chroma_dir if use_persist else None
            )
            st.session_state.vectorstore_exists = True
            st.session_state["last_vectorstore_count"] = len(splitted)
        st.success("✅ Vectorstore ready.")
else:
    if os.path.exists(chroma_dir) and not st.session_state.vectorstore_exists:
        try:
            embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
            vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
            st.session_state.vectorstore_exists = True
            st.info(f"📂 Loaded existing Chroma DB at {chroma_dir}")
        except Exception:
            st.session_state.vectorstore_exists = False

# -------------------------
# Retrieval & Chat
# -------------------------
st.header("2) Retrieval & Chat")

if st.session_state.vectorstore_exists:
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    st.success("🔎 Retrieval engine ready.")
else:
    retriever = None
    st.info("Upload documents first to enable retrieval.")

# Chat UI
if retriever:
    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        with st.spinner("🔎 Retrieving context..."):
            docs = retriever.get_relevant_documents(user_input)

        # Build prompt
        def build_prompt(user_q: str, context_docs: List, max_chars: int = 3000) -> str:
            pieces, total = [], 0
            for d in context_docs:
                text = d.page_content.strip()
                if not text:
                    continue
                if total + len(text) > max_chars:
                    pieces.append(text[: max_chars - total])
                    break
                pieces.append(text)
                total += len(text)
            context_text = "\n\n".join(pieces)
            return f"""You are a helpful assistant. Use the context to answer.

Context:
{context_text}

Question:
{user_q}

Answer clearly and concisely.
"""

        prompt = build_prompt(user_input, docs)

        # Call Groq LLaMA
        def call_llm(prompt_text: str) -> str:
            if groq_api_key and _HAS_GROQ:
                try:
                    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
                    from langchain import LLMChain, PromptTemplate
                    prompt_template = PromptTemplate(input_variables=["text"], template="{text}")
                    chain = LLMChain(llm=llm, prompt=prompt_template)
                    return chain.run({"text": prompt_text}).strip()
                except Exception as e:
                    return f"❌ Groq error: {e}"
            else:
                # fallback
                snippets = [d.page_content.strip() for d in docs[:top_k] if d.page_content.strip()]
                return "(Fallback) " + " ".join(snippets)[:800]

        with st.spinner("🤖 Generating answer..."):
            answer = call_llm(prompt)

        # Save history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", answer))

    # Display chat history
    st.markdown("### 💬 Chat History")
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # Download as PDF
    if st.session_state.chat_history:
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        y = height - 50
        for role, msg in st.session_state.chat_history:
            text = f"{role.upper()}: {msg}"
            for line in text.split("\n"):
                c.drawString(50, y, line)
                y -= 14
                if y < 50:
                    c.showPage()
                    y = height - 50
        c.save()
        pdf_buffer.seek(0)
        st.download_button("⬇️ Download Chat (PDF)", data=pdf_buffer, file_name="chat_history.pdf", mime="application/pdf")

else:
    st.info("⚠️ Please upload documents to enable retrieval and chat.")

st.markdown("---")
st.markdown("**ℹ️ Note:** This app uses Groq LLaMA via API. If no key is provided, fallback summarization is used.")
