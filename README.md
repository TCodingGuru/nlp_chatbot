# NLP Project Chatbot

**Submission Date:** 2025-06-13  
**Team Members:**
- Gilberto Seedorf (655359)

##  Project Overview
This chatbot helps answer questions about our semester-long NLP project. It processes project documents, retrieves contextually relevant information using embeddings, and responds using a locally hosted LLM via Ollama (`llama3`).

##  NLP Techniques Used
- **Text Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Chunking**: Custom word-based splitting
- **Vector Database**: Custom Python implementation with cosine similarity
- **Retrieval-Augmented Generation (RAG)**: Retrieval of top-k chunks + LLM prompt construction
- **LLM**: Local model via Ollama (`llama3`)
- **Context Management**: Persistent chat history with `st.session_state`

##  Know limitations
No advanced context window (uses top-3 chunks only)
Basic prompt structure (can be improved with better instruction tuning)
Single-user history per session
## Setup Instructions
```bash
py -3.8 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install sentence-transformers

streamlit run app.py


