import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from src.vector_db import find_most_similar
from src.chat import chatbot_response

VECTOR_STORE_PATH = "data/vector_store/vector_store.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# --- Loaders with caching ---
@st.cache_data
def load_vector_store():
    with open(VECTOR_STORE_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

# --- Initialize ---
vector_store = load_vector_store()
model = load_model()

# --- App Layout ---
st.title("NLP Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Enter your question:")

# --- Handle special commands ---
if query:
    if query.lower() == "reset":
        st.session_state.history = []
        st.success("Chat history has been reset.")
        st.stop()
    elif query.lower() == "help":
        st.info("Ask any question about the project.\nType 'reset' to start over.")
        st.stop()

    # --- Generate response using Ollama ---
    bot_response, top_chunks, scores, prompt = chatbot_response(query, st.session_state.history, top_k=3)

    # Save history
    st.session_state.history.append({"user": query, "bot": bot_response})

    # Display
    st.markdown("### Top Matching Chunks")
    for idx, chunk in enumerate(top_chunks):
        st.markdown(f"**Chunk {idx + 1}** (Score: {scores[idx]:.3f})")
        st.write(chunk)

    st.markdown("### Assistant Response")
    st.write(bot_response)

    with st.expander("Prompt Used"):
        st.code(prompt, language="markdown")
