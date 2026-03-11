import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    G += np.power(g, 2)
    w -= lr * (1/np.sqrt(G + eps)) * g
    
    return w, G