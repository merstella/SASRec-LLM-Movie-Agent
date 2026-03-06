# Movie Recommendation Project

SASRec-based movie recommendation system with a Groq-powered reasoning agent.

## Project Structure
```text
movie-recommendation/
├─ src/
│  └─ movie_recommendation/
│     ├─ agents/
│     │  ├─ main_agent.py
│     │  └─ agent_utils.py
│     ├─ models/
│     │  └─ model.py
│     ├─ pipelines/
│     │  ├─ convert_parquet.py
│     │  ├─ create_embeddings.py
│     │  ├─ preprocess_sasrec.py
│     │  └─ train.py
│     └─ evaluation/
│        ├─ baseline.py
│        └─ evaluate_sasrec.py
├─ scripts/
│  ├─ run_agent.py
│  ├─ run_baseline.py
│  ├─ run_convert.py
│  ├─ run_embeddings.py
│  ├─ run_eval.py
│  ├─ run_preprocess.py
│  ├─ run_train.py
│  └─ test_tools.py
├─ data/
├─ checkpoints/
├─ logs/
├─ ml-1m/
├─ requirements.txt
└─ .env
```

## Quick Start
```bash
pip install -r requirements.txt
```

Set environment values in `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

## Run Commands
- Convert MovieLens `.dat` files:
```bash
python scripts/run_convert.py
```
- Preprocess for SASRec:
```bash
python scripts/run_preprocess.py
```
- Create item embeddings:
```bash
python scripts/run_embeddings.py
```
- Train SASRec:
```bash
python scripts/run_train.py
```
- Evaluate SASRec:
```bash
python scripts/run_eval.py
```
- Run baselines:
```bash
python scripts/run_baseline.py
```
- Run recommendation agent:
```bash
python scripts/run_agent.py
```
- Tool smoke test:
```bash
python scripts/test_tools.py
```
