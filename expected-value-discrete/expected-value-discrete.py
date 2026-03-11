import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    sum_p = np.sum(p)
    
    if sum_p != 1: 
        raise ValueError
    
    return np.sum(x * p)
    
