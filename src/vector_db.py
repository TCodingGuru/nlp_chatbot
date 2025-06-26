import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-10)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar(query_embedding, embeddings, top_k=5):
    """
    Finds top_k most similar embeddings to the query_embedding.
    
    Args:
        query_embedding: numpy array of shape (embedding_dim,)
        embeddings: numpy array of shape (num_texts, embedding_dim)
        top_k: number of top matches to return
        
    Returns:
        indices: list of top_k indices
        scores: list of similarity scores for top_k
    """
    sims = cosine_similarity(query_embedding.reshape(1, -1), embeddings).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    top_scores = sims[top_indices]
    return top_indices, top_scores
