import pickle
from pathlib import Path
import re

VECTOR_STORE_DIR = Path("data/vector_store")

def load_vector_store(filename="vector_store.pkl"):
    path = VECTOR_STORE_DIR / filename
    with open(path, "rb") as f:
        store = pickle.load(f)
    return store

def clean_text(text):
    # Optionally add your cleaning logic or import from preprocessing.py
    return text.lower().strip()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)                # Normalize whitespace
    text = re.sub(r"[“”\"']", "", text)              # Remove fancy/straight quotes
    text = re.sub(r"[^a-z0-9.,!?()\- ]+", "", text)  # Remove non-alphanumeric (keep basic punctuation)
    text = text.strip()
    return text