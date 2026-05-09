import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, ~6x faster than mpnet on CPU


@st.cache_resource(show_spinner=False)
def _load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def get_embeddings(texts: tuple, model_name: str = MODEL_NAME) -> np.ndarray:
    """Encode a tuple of cleaned texts into dense vectors. Cached by input content."""
    model = _load_model(model_name)
    return model.encode(list(texts), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
