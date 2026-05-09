import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_metrics(
    df_sample: pd.DataFrame,
    labels: np.ndarray,
    cluster_labels: dict,
    compressed: np.ndarray,
) -> pd.DataFrame:
    """Return a DataFrame with avg word count, lexical diversity, and cosine similarity per cluster."""
    rows = []

    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        texts = df_sample["response"][mask].tolist()
        vecs = compressed[mask]

        avg_words = sum(len(t.split()) for t in texts) / len(texts)

        all_words = " ".join(texts).split()
        lexical_div = len(set(all_words)) / len(all_words) if all_words else 0.0

        if len(vecs) > 1:
            sim = cosine_similarity(vecs)
            n = sim.shape[0]
            avg_cosine = (sim.sum() - n) / (n * (n - 1))
        else:
            avg_cosine = 1.0

        rows.append({
            "Cluster": cluster_id,
            "Label": cluster_labels.get(cluster_id, "?"),
            "Size": int(mask.sum()),
            "Avg word count": round(avg_words, 1),
            "Lexical diversity": round(lexical_div, 3),
            "Cosine similarity": round(avg_cosine, 4),
        })

    return pd.DataFrame(rows).set_index("Cluster")
