import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    abs = np.abs(error)

    loss = np.where(abs <= delta, abs * abs / 2, delta * (abs - delta / 2))
    return np.average(loss)
    