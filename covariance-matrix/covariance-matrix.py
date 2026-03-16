import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asmatrix(X, dtype=float)
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    N = len(X)
    if N == 1: 
        return None
    return 1/(N-1) * np.matmul(X_centered.T, X_centered)