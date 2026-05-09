import numpy as np
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_clustering(compressed: np.ndarray) -> tuple:
    """
    Run HDBSCAN and KMeans (k=5,8,10). Return (labels, method_name, silhouette_score)
    for whichever algorithm scores highest.
    """
    # HDBSCAN
    hdb = hdbscan.HDBSCAN(min_cluster_size=5)
    hdb_labels = hdb.fit_predict(compressed)
    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    hdb_score = silhouette_score(compressed, hdb_labels) if n_clusters > 1 else -1.0

    # KMeans — try three values of k
    best_km_score = -1.0
    best_km_labels = None
    best_k = None

    for k in [5, 8, 10]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(compressed)
        score = silhouette_score(compressed, lbl)
        if score > best_km_score:
            best_km_score = score
            best_km_labels = lbl
            best_k = k

    if hdb_score >= best_km_score:
        return hdb_labels, "HDBSCAN", float(hdb_score)
    return best_km_labels, f"KMeans (k={best_k})", float(best_km_score)
