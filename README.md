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
    └── Whizzy.png           # Avatar Picutre  
```

***In Progress***
