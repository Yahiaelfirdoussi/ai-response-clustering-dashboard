import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 20 visually distinct, high-contrast colors — no yellows or near-whites
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#ffffff", "#000000",
]


def build_chart(
    coords: np.ndarray,
    labels: np.ndarray,
    df_sample: pd.DataFrame,
    cluster_labels: dict,
    winning_method: str,
    winning_score: float,
) -> go.Figure:
    """Build the interactive UMAP scatter plot colored by cluster assignment."""
    plot_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": [str(l) for l in labels],
        "response": df_sample["response"].tolist(),
    })

    noise_df = plot_df[plot_df["cluster"] == "-1"]
    cluster_df = plot_df[plot_df["cluster"] != "-1"]

    fig = go.Figure()

    sorted_cids = sorted(cluster_df["cluster"].unique(), key=lambda x: int(x))
    for i, cid in enumerate(sorted_cids):
        sub = cluster_df[cluster_df["cluster"] == cid]
        kw = cluster_labels.get(int(cid), "?")
        color = PALETTE[i % len(PALETTE)]
        hover = (
            "<b>Cluster " + cid + "</b><br>"
            "<b>Keywords:</b> " + kw + "<br><br>"
            "%{customdata}<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            name=f"Cluster {cid} | {kw}",
            marker=dict(size=7, color=color, opacity=0.85, line=dict(width=0.4, color="white")),
            customdata=sub["response"],
            hovertemplate=hover,
        ))

    if len(noise_df) > 0:
        fig.add_trace(go.Scatter(
            x=noise_df["x"],
            y=noise_df["y"],
            mode="markers",
            name="Noise",
            marker=dict(size=4, color="#cccccc", opacity=0.4),
            customdata=noise_df["response"],
            hovertemplate="<b>Noise</b><br><br>%{customdata}<extra></extra>",
        ))

    fig.update_layout(
        title=f"AI Response Clusters — {winning_method} | silhouette: {winning_score:.4f}",
        xaxis_title="UMAP dimension 1",
        yaxis_title="UMAP dimension 2",
        height=650,
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="white",
        margin=dict(l=60, r=320, t=60, b=60),
        legend=dict(
            title="<b>Clusters</b>",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="white",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=11, color="#111111"),
        ),
        hoverlabel=dict(
            bgcolor="#222222",
            font_size=12,
            font_color="white",
            align="left",
        ),
    )

    return fig
