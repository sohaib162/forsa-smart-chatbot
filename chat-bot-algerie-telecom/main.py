from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipelines.guide.guide import run_guide_pipeline
from pipelines.offers.offers import run_offers_pipeline
from pipelines.conventions.conventions import run_conventions_pipeline
from pipelines.depot.depot import run_depot_pipeline

app = FastAPI(title="Forsa Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionBlock(BaseModel):
    categorie_id: Dict[str, str]  # e.g. {"1": "user question"}

class ChatRequest(BaseModel):
    equipe: str
    question: QuestionBlock

CATEGORY_ID_TO_PIPELINE = {
    "1": "offres",
    "2": "conventions",
    "3": "guides",
    "4": "produits",
}

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/process-question")
async def process_question(payload: ChatRequest):
    if not payload.question.categorie_id:
        raise HTTPException(status_code=400, detail="Missing question.categorie_id")

    categorie_id, question = next(iter(payload.question.categorie_id.items()))
    category_name = CATEGORY_ID_TO_PIPELINE.get(str(categorie_id))
    if not category_name:
        raise HTTPException(status_code=400, detail=f"Invalid categorie_id: {categorie_id}")

    if category_name == "offres":
        result = run_offers_pipeline(question)
    elif category_name == "conventions":
        result = run_conventions_pipeline(question)
    elif category_name == "guides":
        result = run_guide_pipeline(question)
    elif category_name == "produits":
        result = run_depot_pipeline(question)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported category: {category_name}")

    answer = (result or {}).get("answer", "")
    sources = (result or {}).get("sources", [])

    return {
        "status": "success",
        "category": category_name,
        "answer": answer,
        "sources": sources,
    }
