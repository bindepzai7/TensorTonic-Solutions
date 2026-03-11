def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    i = len(set(set_a) & set(set_b))
    u = len(set(set_a) | set(set_b))

    if u == 0:
        return 0
    return i/u