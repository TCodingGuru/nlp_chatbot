import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from src.vector_db import find_most_similar

VECTOR_STORE_PATH = "data/vector_store/vector_store.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_data
def load_vector_store():
    with open(VECTOR_STORE_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

vector_store = load_vector_store()
model = load_model()

st.title("NLP Chatbot")

query = st.text_input("Enter your question:")

if query:
    query_emb = model.encode(query)
    embeddings = np.array(vector_store["embeddings"])
    indices, scores = find_most_similar(query_emb, embeddings, top_k=3)
    
    st.write("### Top related chunks:")
    for idx, score in zip(indices, scores):
        st.write(f"**{vector_store['filenames'][idx]}** (score: {score:.3f})")
        st.write(vector_store['texts'][idx])
