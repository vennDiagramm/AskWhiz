# AskWhiz — MMCM Institutional Knowledge Chatbot

AskWhiz is a RAG-based chatbot using **Hybrid RAG + Claude Haiku 4.5 (C9)** — the best-performing configuration identified in the study.

## Project Structure
```
askwhiz/
├── main.py                  # FastAPI backend
├── rag_core.py              # C9 RAG pipeline
├── requirements.txt         # Python dependencies
├── Askwhiz_embeddings.json  # Knowledge base (add this)
├── faiss.index              # FAISS vector index (add this)
└── static/
    └── index.html           # Chat frontend
```

## Setup

### 1. Add your data files
Place these in the root folder:
- `Askwhiz_embeddings.json`
- `faiss.index`

### 2. Set environment variables
```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run locally
```bash
uvicorn main:app --reload
```
Then open http://localhost:8000

## Deploy to Render

1. Push this folder to a GitHub repository
2. Go to https://render.com and create a new **Web Service**
3. Connect your GitHub repo
4. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
6. Deploy!

## API Endpoints
- `GET /` — Chat interface
- `GET /health` — System status
- `POST /ask` — Ask a question
  ```json
  { "question": "When does enrollment begin?" }
  ```
