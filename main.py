"""
main.py
=======
AskWhiz — FastAPI Backend
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time

from rag_core import ask

# ══════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════
app = FastAPI(
    title="AskWhiz API",
    description="Institutional Knowledge Chatbot for Mapúa Malayan Colleges Mindanao",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ══════════════════════════════════════════════════════
# RATE LIMITING
# ══════════════════════════════════════════════════════
request_log: dict = {}
RATE_LIMIT  = 10
RATE_WINDOW = 60

def check_rate_limit(client_ip: str):
    now = time.time()
    if client_ip not in request_log:
        request_log[client_ip] = []
    request_log[client_ip] = [t for t in request_log[client_ip] if now - t < RATE_WINDOW]
    if len(request_log[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment before asking again."
        )
    request_log[client_ip].append(now)

# ══════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer:   str

# ══════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "AskWhiz", "config": "C9 — Hybrid RAG + Claude Haiku 4.5"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: Request, body: QuestionRequest):
    client_ip = request.client.host
    check_rate_limit(client_ip)

    question = body.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Question too long. Please keep it under 500 characters.")

    try:
        result = ask(question)
        return AnswerResponse(
            question=question,
            answer=result,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
