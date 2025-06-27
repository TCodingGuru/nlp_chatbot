# NLP Project Chatbot

**Submission Date:** 27-06-2025  
**Team Members:**
- Gilberto Seedorf (655359)

##  Project Overview
This chatbot helps answer questions about our semester-long NLP project. It processes project documents, retrieves contextually relevant information using embeddings, and responds using a locally hosted LLM via Ollama (`llama3`).

## Architecture Description
The system follows a modular architecture built around the RAG (Retrieval-Augmented Generation) paradigm. Here's an overview of its components:

nlp_chatbot_project/
├── app.py # Main Streamlit app (user interface)
├── requirements.txt
├── README.md
├── data/
│ ├── raw/ # Original project documents
│ ├── processed/ # Preprocessed & chunked text files
│ └── vector_store/ # Stored embeddings for retrieval
├── src/
│ ├── preprocessing.py # Document cleaning & chunking
│ ├── embeddings.py # Text embedding and storage
│ ├── vector_db.py # Custom vector database with similarity search
│ ├── chat.py # RAG pipeline & Ollama LLM interface
│ └── utils.py # Utility functions


### Component Summary:
- **Preprocessing**: Cleans and chunks raw text into 100-word segments for semantic consistency.
- **Embedding**: Converts chunks into dense vectors using `sentence-transformers`.
- **VectorDB**: Stores and retrieves chunks via cosine similarity — implemented from scratch using NumPy.
- **Chat Interface**: Accepts user input, retrieves relevant chunks, builds a prompt, and queries a local LLM (via Ollama).
- **Conversation State**: Maintained using Streamlit session state to enable coherent multi-turn dialogue.
Each layer is decoupled, enabling easy debugging, testing, and future improvements.



##  NLP Techniques Used
- **Text Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Chunking**: Custom word-based splitting
- **Vector Database**: Custom Python implementation with cosine similarity
- **Retrieval-Augmented Generation (RAG)**: Retrieval of top-k chunks + LLM prompt construction
- **LLM**: Local model via Ollama (`llama3`)
- **Context Management**: Persistent chat history with `st.session_state`

##  Known limitations
- No advanced context window (uses top-3 chunks only)  
- Basic prompt structure (can be improved with better instruction tuning)  
- Single-user history per session  
- No evaluation pipeline to assess LLM response quality or retrieval accuracy  
- Limited error handling for missing or malformed vector store files  
- Embeddings are not updated automatically if raw data changes (manual re-run required)  
- Ollama model selection is static (not switchable via UI or config)  
- Streamlit interface lacks advanced UX features (e.g., markdown rendering of LLM output, file upload)


## NLP approach
Our chatbot uses a **Retrieval-Augmented Generation (RAG)** architecture, implemented from scratch, to provide accurate and context-aware responses based on our semester project documents.

### 1. Text Preprocessing & Chunking
- Raw documents are cleaned (lowercased, whitespace normalized).
- Text is split into manageable chunks (~100 words) to preserve semantic coherence.
- Each chunk is saved individually for embedding.

### 2. Embedding Strategy
- We use `sentence-transformers` with the model `all-MiniLM-L6-v2` to convert text chunks and queries into dense vector embeddings.
- This model balances speed and semantic performance, making it suitable for local development.

### 3. Custom Vector Database
- A simple vector store is implemented using Python + NumPy.
- It stores the chunk embeddings, filenames, and raw texts.
- Similarity search is handled using cosine similarity via `scikit-learn`.
- This enables efficient semantic retrieval of the most relevant chunks for a user query.

### 4. Query Handling & RAG Pipeline
- When a user submits a query, we:
  1. Embed the query.
  2. Retrieve the top-k similar text chunks using the vector DB.
  3. Construct a prompt containing: prior conversation (if any), retrieved chunks, and the user query.

### 5. Local LLM Generation with Ollama
- The constructed prompt is sent to a local LLM (e.g., LLaMA3, Mistral, or Phi-3) running via **Ollama**.
- The LLM generates a response using both the retrieved context and conversation history.
- This architecture supports follow-up questions and maintains conversational flow.

### 6. Conversation Management
- Session history is maintained (in memory via Streamlit or CLI context).
- Commands like `"reset"` and `"help"` provide utility and user control.
- Each prompt to the LLM includes this history to enable contextual understanding.

### 7. Why This Approach?
- **Offline-First**: All processing and generation is local — no cloud dependencies.
- **Transparent NLP**: We avoided black-box libraries (e.g., LangChain) to fully implement and understand every RAG step.
- **Modular Design**: Each stage (preprocessing, embeddings, vector DB, LLM) is modular and testable.

## Prerequisites
**Ollama Local LLM**
- Download and install the Ollama app from https://ollama.com/download.
- Launch the Ollama app (it runs a local server at http://localhost:11434)
```bash
- install ollama pull llama3
```

## Setup Instructions
```bash
py -3.8 -m venv .venv    # project incompatibility with python 3.9 and higher
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install sentence-transformers

streamlit run app.py
