from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from movie_recommendation.agents.main_agent import GROQ_MODEL, invoke_agent

app = FastAPI(
    title="Movie Recommendation API",
    version="1.0.0",
    description="SASRec + query rerank + Groq reason generation",
)


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    query: str = Field(default="")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "movie-recommendation-api",
        "llm_model": GROQ_MODEL,
    }


@app.post("/recommend")
def recommend(payload: RecommendRequest):
    try:
        return invoke_agent(user_id=payload.user_id, user_query=payload.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
