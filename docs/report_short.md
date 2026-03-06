# Short Report

## 1) Architecture
The system combines sequential recommendation with query-aware reranking and LLM explanation:

1. SASRec generates top candidate movies from user behavior sequence.
2. Sentence-transformer reranks candidates by cosine similarity to the query.
3. Groq LLM formats recommendation reasons into structured JSON.
4. FastAPI and Streamlit provide deployable backend and demo frontend.

## 2) Attention Equations
SASRec attention core:

\[
Q = XW_Q,\ K = XW_K,\ V = XW_V
\]

\[
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
\]

where `M` is a causal mask to prevent attending to future items.

## 3) Dataset and Pipeline
Dataset: MovieLens 1M.

Pipeline:
1. Convert raw data (`run_convert.py`)
2. Preprocess and split sequences (`run_preprocess.py`)
3. Build item embeddings (`run_embeddings.py`)
4. Train SASRec (`run_train.py`)
5. Evaluate SASRec and baselines (`run_eval.py`, `run_baseline.py`)

## 4) Metrics Table
| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity | TBD | TBD |
| Item-KNN | TBD | TBD |
| SASRec | TBD | TBD |

## 5) Deployment and Demo
FastAPI:
- `GET /health`
- `POST /recommend`

Streamlit:
- Input: `user_id`, query text
- Output: movie list with reasons
