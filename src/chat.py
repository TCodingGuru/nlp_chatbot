# src/chat.py

from sentence_transformers import SentenceTransformer
import numpy as np
from src.utils import load_vector_store
from src.vector_db import find_most_similar
import ollama

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

OLLAMA_MODEL = "llama3"  # or mistral, phi3, etc.

def build_prompt(history, context_chunks, user_input):
    """
    Build a prompt using history and context for the LLM.
    """
    prompt = "You are an expert assistant helping answer questions about a student NLP project.\n"

    if history:
        prompt += "\nConversation History:\n"
        for turn in history:
            prompt += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

    prompt += "\nProject Context:\n"
    for chunk in context_chunks:
        prompt += f"{chunk}\n"

    prompt += f"\nUser: {user_input}\nAssistant:"
    return prompt

def generate_response_with_ollama(prompt, model=OLLAMA_MODEL):
    """
    Sends the prompt to Ollama and returns the generated response.
    """
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])
    return response["message"]["content"]

def chatbot_response(user_input, history=None, top_k=3):
    store = load_vector_store()
    texts = store["texts"]
    embeddings = np.array(store["embeddings"])

    user_emb = model.encode([user_input])[0]
    indices, scores = find_most_similar(user_emb, embeddings, top_k)

    top_chunks = [texts[i] for i in indices]
    prompt = build_prompt(history, top_chunks, user_input)

    response = generate_response_with_ollama(prompt)

    return response, top_chunks, scores, prompt
