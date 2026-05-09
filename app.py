import numpy as np
import pandas as pd
import streamlit as st

from core.autoencoder import train_autoencoder
from core.clusterer import run_clustering
from core.embedder import MODEL_NAME, get_embeddings
from core.labeler import get_cluster_labels
from core.metrics import compute_metrics
from core.reducer import reduce_to_2d
from ui.chart import build_chart
from ui.sidebar import render_results, render_settings
from utils.preprocessor import preprocess_df

st.set_page_config(
    page_title="AI Response Clustering Dashboard",
    page_icon="🔍",
    layout="wide",
)

# ── Cached wrappers for slow pipeline steps ────────────────────────────────────
# These live in app.py (not core/) so core modules stay free of Streamlit imports.

@st.cache_data(show_spinner=False)
def _compress(embeddings: np.ndarray) -> np.ndarray:
    return train_autoencoder(embeddings)


@st.cache_data(show_spinner=False)
def _cluster(compressed: np.ndarray):
    return run_clustering(compressed)


@st.cache_data(show_spinner=False)
def _umap(compressed: np.ndarray) -> np.ndarray:
    return reduce_to_2d(compressed)


# ── Page ───────────────────────────────────────────────────────────────────────
st.title("AI Response Clustering Dashboard")
st.markdown(
    "Upload AI responses to automatically discover topic clusters using "
    f"sentence embeddings (`{MODEL_NAME}`) → autoencoder → HDBSCAN / KMeans → UMAP."
)

config = render_settings()
sample_size = config["sample_size"]

uploaded = st.file_uploader(
    "Upload a **CSV** (needs a `response` column) "
    "or a **Parquet** file (Chatbot Arena format — extracts assistant turns automatically)",
    type=["csv", "parquet"],
)

if uploaded is None:
    st.info("Upload a file above to start the clustering pipeline.")
    st.stop()

# ── Load & sample ──────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    if uploaded.name.endswith(".parquet"):
        raw_df = pd.read_parquet(uploaded)

        def _first_assistant(conversation):
            for turn in conversation:
                if turn["role"] == "assistant":
                    return turn["content"]
            return None

        raw_df["response"] = raw_df["conversation_a"].apply(_first_assistant)
        df = (
            raw_df[raw_df["language"] == "English"][["response"]]
            .dropna()
            .reset_index(drop=True)
        )
    else:
        df = pd.read_csv(uploaded)
        if "response" not in df.columns:
            st.error("The CSV must contain a `response` column.")
            st.stop()
        df = df[["response"]].dropna().reset_index(drop=True)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    df = preprocess_df(df)

st.success(f"Loaded **{len(df):,}** responses — ready to cluster.")

# ── Pipeline (each step cached after first run) ────────────────────────────────
with st.spinner(f"Encoding with `{MODEL_NAME}` (384-dim)…"):
    embeddings = get_embeddings(tuple(df["clean"].tolist()))

with st.spinner("Training autoencoder: 384 → 64 dims…"):
    compressed = _compress(embeddings)

with st.spinner("Clustering: HDBSCAN + KMeans (k = 5, 8, 10)…"):
    labels, winning_method, winning_score = _cluster(compressed)

with st.spinner("UMAP 2D reduction…"):
    coords = _umap(compressed)

with st.spinner("Extracting TF-IDF cluster keywords…"):
    cluster_labels = get_cluster_labels(df["clean"], labels)

# ── Sidebar results ────────────────────────────────────────────────────────────
render_results(winning_method, winning_score, cluster_labels)

# ── Chart ──────────────────────────────────────────────────────────────────────
metrics_df = compute_metrics(df, labels, cluster_labels, compressed)

st.markdown("---")
st.subheader("Cluster map")
fig = build_chart(coords, labels, df, cluster_labels, winning_method, winning_score)
st.plotly_chart(fig, use_container_width=False)

# ── Metrics ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Per-cluster quality metrics")
st.dataframe(metrics_df, use_container_width=True)
