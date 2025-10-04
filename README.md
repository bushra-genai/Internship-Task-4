# Internship-Task-4
📚 RAG Q/A — Retrieval Augmented Generation (Groq + LLaMA)

A Streamlit-based RAG (Retrieval-Augmented Generation) application that allows users to upload PDF/TXT documents, build embeddings with HuggingFace models, store them in a Chroma vector database, and perform intelligent Q/A using Groq’s LLaMA model.
If no API key is provided, the app falls back to a summarization-based retrieval mode.

✨ Features

📂 Upload and process multiple PDF/TXT documents

🧩 Chunking of documents with customizable size and overlap

🔎 Vector search with HuggingFace embeddings + ChromaDB

🤖 Q/A powered by Groq’s LLaMA (or fallback mode without API key)

💬 Chat UI with history for interactive document exploration

⬇️ Export chat history as PDF

⚙️ Configurable sidebar settings (top-k retrieval, chunk size, overlap, etc.)

🛠️ Tech Stack

Frontend: Streamlit

Embeddings: HuggingFace Sentence Transformers

Vector Store: ChromaDB

LLM Backend: Groq LLaMA
 via langchain-groq

PDF Export: ReportLab

📂 Project Structure
app_rag_streamlit.py   # Main Streamlit app
requirements.txt       # Dependencies
chroma_db/             # (Optional) Persistent Chroma vector database

🚀 Installation

Clone the repository

git clone https://github.com/bushra-genai/Internship-Task-4
cd rag-streamlit-app


Create virtual environment (recommended)

python -m venv rag_env
rag_env\Scripts\activate   # Windows
# OR
source rag_env/bin/activate   # Mac/Linux


Install dependencies

pip install -r requirements.txt

▶️ Usage

Run the app

streamlit run app_rag_streamlit.py


Upload documents (PDF/TXT) from the UI

Ask questions in the chat box — the app retrieves relevant chunks and answers using Groq LLaMA (if API key is provided).

Export chat history as PDF for saving or sharing.

🔑 Environment Variables

Create a .env file in the project root and add your Groq API key:

GROQ_API_KEY=your_groq_api_key_here

📸 Screenshots

(Add your app screenshots here to make the README visually appealing.)

🙌 Author

👩‍💻 Developed by Bushra Mubeen

📜 License

This project is licensed under the MIT License — you’re free to use, modify, and distribute with attribution.
