import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def get_cluster_labels(
    clean_texts: pd.Series,
    labels: np.ndarray,
    top_n: int = 5,
) -> dict:
    """Return {cluster_id: 'kw1, kw2, ...'} using TF-IDF top keywords per cluster."""
    cluster_labels = {}

    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            cluster_labels[-1] = "Noise"
            continue

        texts = clean_texts[labels == cluster_id].tolist()
        vectorizer = TfidfVectorizer(max_features=top_n, stop_words="english")
        try:
            vectorizer.fit(texts)
            cluster_labels[cluster_id] = ", ".join(vectorizer.get_feature_names_out())
        except ValueError:
            cluster_labels[cluster_id] = "unknown"

    return cluster_labels
