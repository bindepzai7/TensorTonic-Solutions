import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    def norm(x):
        return np.sqrt(np.sum(np.power(x, 2)))
    
    def cosine(x1, x2):
        return np.sum(x1 * x2) / (norm(x1) * norm(x2))

    if label == 1:
        return 1 - cosine(x1, x2)

    return max(0, cosine(x1, x2) - margin)