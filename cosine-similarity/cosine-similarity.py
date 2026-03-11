import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)

    if norm_a * norm_b == 0: 
        return 0.0

    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))