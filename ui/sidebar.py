import streamlit as st


def render_settings() -> dict:
    """Render the top-of-sidebar controls. Call this BEFORE running the pipeline."""
    st.sidebar.title("Settings")
    sample_size = st.sidebar.slider(
        "Sample size",
        min_value=500,
        max_value=10_000,
        value=500,
        step=500,
        help="Number of responses to sample from the uploaded dataset.",
    )
    return {"sample_size": sample_size}


def render_results(winning_method: str, winning_score: float, cluster_labels: dict) -> None:
    """Render clustering results in the sidebar. Call this AFTER the pipeline runs."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Clustering Results")
    st.sidebar.metric("Winner", winning_method)
    st.sidebar.metric("Silhouette score", f"{winning_score:.4f}")

    st.sidebar.markdown("### Cluster Labels")
    for cid in sorted(cluster_labels):
        prefix = "Noise" if cid == -1 else f"Cluster {cid}"
        st.sidebar.markdown(f"**{prefix}** → {cluster_labels[cid]}")
