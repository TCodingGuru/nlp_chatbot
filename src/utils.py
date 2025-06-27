import pickle
from pathlib import Path

VECTOR_STORE_DIR = Path("data/vector_store")

def load_vector_store(filename="vector_store.pkl"):
    path = VECTOR_STORE_DIR / filename
    with open(path, "rb") as f:
        store = pickle.load(f)
    return store

def clean_text(text):
    # Optionally add your cleaning logic or import from preprocessing.py
    return text.lower().strip()
