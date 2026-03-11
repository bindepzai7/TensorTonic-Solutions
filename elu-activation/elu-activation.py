import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    res = [s if s > 0 else alpha * (math.exp(s) - 1) for s in x]
    return res