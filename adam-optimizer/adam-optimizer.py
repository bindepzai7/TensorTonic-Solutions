import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param = np.asarray(param, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)
    
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * np.square(grad)

    m_hat = m_new / (1 - np.power(beta1, t))
    v_hat = v_new / (1 - np.power(beta2, t))

    param = param - lr * (m_hat / (np.sqrt(v_hat) + eps))

    return param, m_new, v_new