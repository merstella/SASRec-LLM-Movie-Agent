# Movie Recommendation System

End-to-end movie recommendation project based on SASRec, semantic reranking, Groq LLM reasoning, FastAPI deployment, and Streamlit demo.

## One-Click Deploy (No Local Install)
This repo now includes Render blueprint config at `render.yaml`.

Deploy directly from GitHub: [click here](https://render.com/deploy?repo=https://github.com/merstella/SASRec-LLM-Movie-Agent)

On Render:
1. Click the deploy link above.
2. Keep the detected `render.yaml` service.
3. Set required secrets: `GROQ_API_KEY` and `TMDB_API_KEY`.
4. Click **Create New Web Service**.
5. Use the generated public URL to share your live demo (users only open link, no installation needed).

## Project Structure
```text
movie-recommendation/
├─ src/
│  └─ movie_recommendation/
│     ├─ agents/
│     │  ├─ main_agent.py
│     │  └─ agent_utils.py
│     ├─ api/
│     │  └─ app.py
│     ├─ evaluation/
│     │  ├─ baseline.py
│     │  └─ evaluate_sasrec.py
│     ├─ models/
│     │  └─ model.py
│     └─ pipelines/
│        ├─ convert_parquet.py
│        ├─ create_embeddings.py
│        ├─ preprocess_sasrec.py
│        └─ train.py
├─ scripts/
│  ├─ run_agent.py
│  ├─ run_api.py
│  ├─ run_baseline.py
│  ├─ run_convert.py
│  ├─ run_embeddings.py
│  ├─ run_eval.py
│  ├─ run_preprocess.py
│  ├─ run_train.py
│  ├─ streamlit_demo.py
│  └─ test_tools.py
├─ docs/
│  └─ report_short.md
├─ data/
├─ checkpoints/
├─ logs/
├─ ml-1m/
├─ requirements.txt
└─ .env
```

## Architecture
1. Candidate generation: SASRec predicts next likely items from user sequence history.
2. Query-aware rerank: sentence-transformer embedding reranks SASRec top-`k` candidates by cosine similarity to the user query.
3. Reason generation: Groq LLM converts recommendation list into user-facing reasons in strict JSON format.
4. Serving:
   - FastAPI for `/health` and `/recommend`
   - Streamlit for interactive demo

## Attention Equations
SASRec uses causal self-attention over user history sequence.

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

- `X`: input sequence embeddings (item + position)
- `M`: causal mask that blocks future positions
- `d_k`: key dimension scaling factor

## Dataset and Pipeline
Dataset: MovieLens 1M (`ml-1m`).

Pipeline:
1. `run_convert.py`: convert raw `.dat` files to parquet (`data/interactions.parquet`, `data/items.parquet`).
2. `run_preprocess.py`: map IDs and split sequence data into train/val/test (`data/sasrec_data.pkl`).
3. `run_embeddings.py`: build item text embeddings (`data/item_emb.npy`, `data/item_meta.json`).
4. `run_train.py`: train SASRec and save checkpoint (`checkpoints/sasrec.pt`).
5. `run_eval.py`: evaluate SASRec Recall@10 and NDCG@10.
6. `run_baseline.py`: evaluate Popularity and Item-KNN baselines.

## Metrics
Current logs in `logs/` include training loss curves.  
Run `run_eval.py` and `run_baseline.py` to populate final benchmark table.

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity | 0.0353 | 0.0174 |
| Item-KNN | 0.0631 | 0.0334 |
| **SASRec** | **0.2098** | **0.1111**|

## Phase 4 Deploy + Demo
### FastAPI
- `GET /health`
- `POST /recommend` with body:
```json
{ "user_id": 5, "query": "action movies with space elements" }
```

Run:
```bash
python scripts/run_api.py
```

### Streamlit Demo
Input:
- `user_id`
- free-text `query`

Output:
- movie recommendations
- generated reason per movie

Run:
```bash
streamlit run scripts/streamlit_demo.py
```

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

Set environment values in `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
TMDB_API_KEY=your_tmdb_api_key_here
```

## Quick Run Commands
```bash
python scripts/run_convert.py
python scripts/run_preprocess.py
python scripts/run_embeddings.py
python scripts/run_train.py
python scripts/run_eval.py
python scripts/run_baseline.py
python scripts/run_agent.py
```
