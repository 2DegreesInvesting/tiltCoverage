from Levenshtein import distance, jaro_winkler
from thefuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


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
    # Lower bound of 0 (two strings are identical)
    upper_bound = max(len(s1), len(s2))
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
