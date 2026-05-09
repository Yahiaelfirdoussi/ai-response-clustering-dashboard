# AI Response Clustering Dashboard

A fully local Streamlit app that automatically discovers topic clusters in AI-generated responses using a multi-stage NLP pipeline — no cloud services required.

---

## Demo

Upload any dataset of AI responses and get an interactive 2D cluster map with per-cluster keyword labels, quality metrics, and hover-to-read functionality.

---

## How it works

```
Raw text
   │
   ▼
Preprocessing        lowercase · remove special chars · strip whitespace
   │
   ▼
Sentence Embeddings  all-MiniLM-L6-v2  →  384-dim vectors
   │
   ▼
Autoencoder          PyTorch  384 → 64 dims  (50 epochs, MSE loss)
   │
   ▼
Clustering           HDBSCAN  vs  KMeans (k = 5, 8, 10)
                     winner picked by silhouette score
   │
   ▼
UMAP                 64 dims → 2D  (visualization only)
   │
   ▼
TF-IDF Labels        top 5 keywords per cluster
   │
   ▼
Plotly Chart         interactive scatter plot  +  per-cluster metrics table
```

---

## Project structure

```
├── app.py                  # Streamlit entry point
├── core/
│   ├── embedder.py         # Sentence-transformer encoding (cached)
│   ├── autoencoder.py      # PyTorch autoencoder definition + training
│   ├── clusterer.py        # HDBSCAN + KMeans with silhouette selection
│   ├── reducer.py          # UMAP 2D reduction
│   ├── labeler.py          # TF-IDF keyword extraction per cluster
│   └── metrics.py          # Per-cluster quality metrics
├── ui/
│   ├── chart.py            # Plotly interactive scatter plot
│   └── sidebar.py          # Streamlit sidebar controls + results
├── utils/
│   └── preprocessor.py     # Text cleaning and parquet loading
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/Yahiaelfirdoussi/ai-response-clustering-dashboard.git
cd ai-response-clustering-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) and upload your data file.

---

## Input format

The app accepts two file types:

| Format | Requirement |
|--------|-------------|
| **CSV** | Must have a `response` column — one AI response per row |
| **Parquet** | Chatbot Arena format — `conversation_a` column with `[{role, content}]` turns; assistant responses are extracted automatically |

---

## Sidebar controls

- **Sample size** — number of responses to process (default 500, max 10 000)

Results shown after the pipeline runs:
- Winning clustering method and silhouette score
- TF-IDF keyword label for each cluster

---

## Dependencies

```
streamlit
sentence-transformers==3.0.1
transformers==4.41.2
torch
hdbscan
umap-learn
scikit-learn
plotly
pandas
numpy
```

---

## Author

**Yahya Elfirdoussi** — Data Scientist & ML Engineer  
[LinkedIn](https://linkedin.com/in/yahya-elfirdoussi) · [Portfolio](https://yahiaelfirdoussi.netlify.app) · [GitHub](https://github.com/yahiaelfirdoussi)
