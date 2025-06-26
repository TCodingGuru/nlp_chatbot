# src/chat.py

from sentence_transformers import SentenceTransformer
import numpy as np
from utils import load_vector_store
from vector_db import find_most_similar

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def chatbot_response(user_input, top_k=3):
    store = load_vector_store()
    texts = store["texts"]
    embeddings = np.array(store["embeddings"])

    user_emb = model.encode([user_input])[0]
    indices, scores = find_most_similar(user_emb, embeddings, top_k)

    responses = [texts[i] for i in indices]
    return responses, scores

if __name__ == "__main__":
    print("Welcome to the simple NLP chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        responses, scores = chatbot_response(user_input)
        print("\nBot top responses:")
        for i, (resp, score) in enumerate(zip(responses, scores), 1):
            print(f"{i}. (score: {score:.4f}) {resp}")
