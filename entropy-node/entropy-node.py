import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y).astype(int)
    counts = np.bincount(y)
    probs = counts / np.sum(counts)
    log_probs = np.where(probs > 0, np.log2(probs), 0)
    return -np.dot(probs, log_probs)