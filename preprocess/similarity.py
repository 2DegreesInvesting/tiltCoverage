from Levenshtein import distance, jaro_winkler
from thefuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List

import numpy as np


def get_levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate similarity ratio  between two strings based on the
    Levenshtein distance

    Args:
        s1 (str): First string to compare
        s2 (str): Second string to compare

    Returns:
        float: Similarity score between 0.0 (not the same) and 1.0 (identical)
    """

    # Levenshtein has the upper bound of the longest string of the two strings
    # and lower bound of 0 (two strings are identical)
    upper_bound = max(len(s1), len(s2))

    # distance is integer
    levenshtein = distance(s1, s2)

    # take the ratio and subtract from one to denote similarity
    return 1 - (levenshtein / upper_bound)


def get_jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro Winkler similarity between two strings

    Args:
        s1 (str): First string to compare
        s2 (str): Second string to compare

    Returns:
        float: Similarity score between (not the same) and 1.0 (identical)
    """

    return jaro_winkler(s1, s2)


def get_fuzzy_similarity(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings based on fuzzy string matching.

    Args:
        s1 (str): First string to compare
        s2 (str): Second string to compare

    Returns:
        float: Similarity score between (not the same) and 1.0 (identical)
    """

    # fuzzy matching returns value between 0 and 100
    return fuzz.ratio(s1, s2) / 100


def get_cosine_similarity(
    s1: str, s2: str, model: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> float:
    """Calculate the cosine similarity between the embeddings of two strings.

    Args:
        s1 (str): First string to compare
        s2 (str): Second string to compare
        model (str, optional): SentenceTransformer model to embed the strings. Defaults to "paraphrase-multilingual-MiniLM-L12-v2".

    Returns:
        float: Similarity score between (not the same) and 1.0 (identical)
    """

    embedder = SentenceTransformer(model)
    embeddings = embedder.encode([s1, s2])
    e1, e2 = embeddings[0], embeddings[1]

    return cosine_similarity(e1, e2)


def get_weighted_similarity_scores(s1: str, s2: str, weights: List[float]) -> float:
    """Calculate weighted similarity score between two strings.

    Args:
        s1 (str): First string to compare
        s2 (str): Second string to compare
        weights (List[float]): Weights to assign to the different similarity scores, each value in the list must be between 0 and 1, and the values of the list must add up to 1. [Levenshtein, Jaro Winkler, Fuzzy, Cosine].

    Returns:
        float: Weighted similarity score.
    """
    assert sum(weights) == 1, "Sum of weights do not add up to 1"

    l = get_levenshtein_similarity(s1, s2)
    j = get_jaro_winkler_similarity(s1, s2)
    f = get_fuzzy_similarity(s1, s2)
    c = get_cosine_similarity(s1, s2)

    return np.matmul([l, j, f, c], weights)
