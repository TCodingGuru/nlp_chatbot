# src/embeddings.py

from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

PROCESSED_DIR = Path("data/processed")
VECTOR_STORE_DIR = Path("data/vector_store")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Load the embedding model (local transformer)
MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast, and good enough
model = SentenceTransformer(MODEL_NAME)

def get_text_chunks():
    chunks = []
    for file_path in PROCESSED_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            chunks.append((file_path.name, text))
    return chunks

def embed_and_store():
    chunks = get_text_chunks()
    texts = [text for _, text in chunks]
    embeddings = model.encode(texts)

    # Save to disk as a pickle file
    store = {
        "filenames": [f for f, _ in chunks],
        "texts": texts,
        "embeddings": embeddings.tolist()  # Convert from numpy to list
    }

    with open(VECTOR_STORE_DIR / "vector_store.pkl", "wb") as f:
        pickle.dump(store, f)

    print(f"Stored {len(texts)} embeddings to vector_store.pkl")

if __name__ == "__main__":
    embed_and_store()
