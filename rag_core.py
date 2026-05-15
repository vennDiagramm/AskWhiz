"""
rag_core.py
===========
AskWhiz — C9 RAG Pipeline
Retrieval : Hybrid (Dense FAISS + Sparse BM25) / OpenSearch (for future scalability)
LLM       : Claude Haiku 4.5 (Anthropic)
"""

import os
import json
import time
import numpy as np
import faiss
from opensearchpy import OpenSearch
from typing import List, Tuple
from rank_bm25 import BM25Okapi          # pip install rank-bm25

from openai import OpenAI
import anthropic

from dotenv import load_dotenv
load_dotenv()



# ══════════════════════════════════════════════════════
# CONFIG — set these as environment variables in production
# ══════════════════════════════════════════════════════
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

EMBED_MODEL         = "text-embedding-3-small"
DATA_FILE           = "Askwhiz_embeddings.json"
INDEX_FILE          = "faiss.index"
ANTHROPIC_GEN_MODEL = "claude-haiku-4-5-20251001"
TOP_K               = 5
BM25_WEIGHT         = 0.4   # 40% BM25, 60% dense

OPENSEARCH_URL   = os.environ.get("OPENSEARCH_URL", "")
OPENSEARCH_USER  = os.environ.get("OPENSEARCH_USER", "")
OPENSEARCH_PASS  = os.environ.get("OPENSEARCH_PASS", "")
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "")

openai_client    = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ══════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════
def load_documents(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["chunk_text"] for item in data]

documents = load_documents(DATA_FILE)
print(f"[INIT] Loaded {len(documents)} document chunks.")


"""

# ══════════════════════════════════════════════════════
# FAISS INDEX (Dense)
# ══════════════════════════════════════════════════════
def build_or_load_index() -> faiss.Index:
    if os.path.exists(INDEX_FILE):
        print("[INIT] Loading existing FAISS index...")
        return faiss.read_index(INDEX_FILE)

    print("[INIT] Building FAISS index from embeddings...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = np.array([item["embedding"] for item in data], dtype="float32")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"[INIT] FAISS index built with {index.ntotal} vectors.")
    return index

faiss_index = build_or_load_index()

"""

# ══════════════════════════════════════════════════════
# OPENSEARCH CONNECTION AND INDEX
# ══════════════════════════════════════════════════════
opensearch_client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,
    verify_certs=True,
    timeout=30
)
print("[INIT] Connected to OpenSearch")

def build_or_load_opensearch_index():
    """
    If the index already exists and has data → use it.
    If not → create it and ingest from AskWhiz_embeddings.json.
    Mirrors the FAISS build_or_load_index() behavior.
    """
    # Check if index exists and has documents
    if opensearch_client.indices.exists(index=OPENSEARCH_INDEX):
        count = opensearch_client.count(index=OPENSEARCH_INDEX)["count"]
        if count > 0:
            print(f"[INIT] Loading existing OpenSearch index ({count} docs)...")
            return
        else:
            print("[INIT] Index exists but is empty. Re-ingesting...")
            opensearch_client.indices.delete(index=OPENSEARCH_INDEX)

    # Create fresh index
    print("[INIT] Creating OpenSearch index...")
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                "chunk_id"      : {"type": "keyword"},
                "source"        : {"type": "text"},
                "document_type" : {"type": "keyword"},
                "audience"      : {"type": "keyword"},
                "title"         : {"type": "text"},
                "chunk_text"    : {"type": "text"},
                "token_count"   : {"type": "integer"},
                "embedding"     : {
                    "type"      : "knn_vector",
                    "dimension" : 1536,
                    "method"    : {
                        "name"       : "hnsw",
                        "space_type" : "cosinesimil",
                        "engine"     : "faiss"
                    }
                }
            }
        }
    }
    opensearch_client.indices.create(index=OPENSEARCH_INDEX, body=index_body)
    print(f"[INIT] Index created: {OPENSEARCH_INDEX}")

    # Ingest embeddings
    print("[INIT] Ingesting AskWhiz_embeddings.json...")
    with open("AskWhiz_embeddings.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[INIT] Loaded {len(chunks)} chunks — ingesting now...")
    success = 0
    failed  = 0

    for chunk in chunks:
        doc = {
            "chunk_id"      : chunk["chunk_id"],
            "source"        : chunk["source"],
            "document_type" : chunk.get("document_type", "handbook"),
            "audience"      : chunk["audience"],
            "title"         : chunk["title"],
            "chunk_text"    : chunk["chunk_text"],
            "token_count"   : chunk["token_count"],
            "embedding"     : chunk["embedding"]
        }
        try:
            opensearch_client.index(
                index=OPENSEARCH_INDEX,
                id=chunk["chunk_id"],
                body=doc
            )
            success += 1
        except Exception as e:
            print(f"  ❌ Failed: {chunk['chunk_id']} — {e}")
            failed += 1

    time.sleep(2)  # wait for index to refresh
    count = opensearch_client.count(index=OPENSEARCH_INDEX)["count"]
    print(f"[INIT] Ingestion complete — {count} docs in OpenSearch")

# To know if OpenSearch is available at startup

try:
    build_or_load_opensearch_index()
except Exception as e:
    print(f"[WARN] OpenSearch unavailable at startup: {e}")
    print("[WARN] App will start, but dense retrieval may fail until OpenSearch is reachable.")

# ══════════════════════════════════════════════════════
# BM25 INDEX (Sparse)
# ══════════════════════════════════════════════════════
bm25_index = BM25Okapi([doc.lower().split() for doc in documents])
print("[INIT] BM25 index built.")

# ══════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════
def embed_query(query: str) -> list:
    result = openai_client.embeddings.create(model=EMBED_MODEL, input=[query])
    return result.data[0].embedding


def dense_retrieve(query: str, top_k: int) -> List[Tuple[str, float]]:
    """Returns list of (chunk_id, score) from OpenSearch kNN."""
    query_vector = embed_query(query)
    search_body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k"     : top_k
                }
            }
        }
    }
    response = opensearch_client.search(index=OPENSEARCH_INDEX, body=search_body)
    return [
        (hit["_id"], hit["_score"])
        for hit in response["hits"]["hits"]
    ]

""""
def dense_retrieve(query: str, top_k: int) -> List[Tuple[int, float]]:
    query_vec = embed_query(query)
    scores, indices = faiss_index.search(query_vec, top_k)
    return list(zip(indices[0].tolist(), scores[0].tolist()))
"""

def sparse_retrieve(query: str, top_k: int) -> List[Tuple[int, float]]:
    scores = bm25_index.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]

def normalize(results: List[Tuple[int, float]]) -> dict:
    if not results:
        return {}
    scores = [s for _, s in results]
    min_s, max_s = min(scores), max(scores)
    span = max_s - min_s if max_s != min_s else 1.0
    return {idx: (s - min_s) / span for idx, s in results}

def hybrid_retrieve(query: str, top_k: int = TOP_K) -> List[str]:
    dense_results  = dense_retrieve(query, top_k * 2)
    sparse_results = sparse_retrieve(query, top_k * 2)

    dense_norm  = normalize(dense_results)
    sparse_norm = normalize(sparse_results)

    # Convert dense chunk_ids back to int indices for fusion
    chunk_id_to_idx = {}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, item in enumerate(data):
        chunk_id_to_idx[item["chunk_id"]] = i

    # Remap dense_norm keys from chunk_id strings → int indices
    dense_norm_remapped = {
        chunk_id_to_idx[k]: v
        for k, v in dense_norm.items()
        if k in chunk_id_to_idx
    }

    all_indices = set(dense_norm_remapped.keys()) | set(sparse_norm.keys())
    fused = {
        idx: (1 - BM25_WEIGHT) * dense_norm_remapped.get(idx, 0.0)
             + BM25_WEIGHT * sparse_norm.get(idx, 0.0)
        for idx in all_indices
    }

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print("\n--- Retrieved Chunks (Hybrid) ---")
    for rank, (idx, score) in enumerate(ranked):
        print(f"[{rank+1}] (fused score: {score:.3f}) {documents[idx][:200]}")
    print("---------------------------------\n")

    return [documents[idx] for idx, _ in ranked]

# ══════════════════════════════════════════════════════
# CHAIN-OF-THOUGHT PROMPT
# ══════════════════════════════════════════════════════
SYSTEM_PERSONA = (
    "You are AskWhiz, the official AI assistant for Mapúa Malayan Colleges Mindanao (MMCM). "
    "You help students, faculty, and staff by answering questions based strictly on the provided context. "
    "You are accurate, concise, and professional."
)

def build_cot_prompt(question: str, chunks: List[str]) -> str:
    context = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        f"{SYSTEM_PERSONA}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer ONLY using the provided context below.\n"
        "- If the answer is not found in the context, respond with: "
        "'I do not have that information in the MMCM handbook.'\n"
        "- Do NOT fabricate, assume, or use outside knowledge.\n"
        "- Think step-by-step before giving your final answer.\n\n"
        "FORMAT YOUR RESPONSE AS:\n"
        "A clear, direct, and natural answer as if you are a knowledgeable assistant. "
        "Do not mention chunks, reasoning steps, or your thought process. "
        "Do not use labels like 'Reasoning:' or 'Answer:'. "
        "You may reference 'the MMCM handbook' or 'MMCM policy' naturally when relevant.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
    )
"""
        <The above prompt is designed to elicit a direct answer without explicit reasoning steps.>
        <This is done to make C9's response more flowing and natural, while still grounding it in the retrieved context.>
        <Unlike the testing process where we want explicit reasoning for RAGAS evaluation, the live API should prioritize a user-friendly answer.>
        
        "FORMAT YOUR RESPONSE AS:\n"
        "Reasoning: <briefly explain what in the context supports your answer>\n"
        "Answer: <your clear, direct answer>\n\n"
"""


# ══════════════════════════════════════════════════════
# GENERATION (Anthropic / C9)
# ══════════════════════════════════════════════════════
def generate(prompt: str) -> str:
    while True:
        try:
            message = anthropic_client.messages.create(
                model=ANTHROPIC_GEN_MODEL,
                max_tokens=1024,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print("[Anthropic] Rate limited — waiting 15s...")
                time.sleep(15)
            else:
                raise

# ══════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════
def ask(question: str) -> str:
    """
    Main entry point. Takes a question string,
    runs the C9 Hybrid RAG pipeline,
    and returns the answer string.
    """
    print("[Config: C9] Retrieval=HYBRID  LLM=ANTHROPIC")
    chunks = hybrid_retrieve(question)
    prompt = build_cot_prompt(question, chunks)
    return generate(prompt)

def run_batch(test_dataset_path: str, output_path: str):
    """
    Runs all questions in the test dataset through the C9 RAG pipeline
    and saves results to a JSON file for RAGAS evaluation.
    """
    with open(test_dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    total = len(test_data)

    for i, item in enumerate(test_data):
        question     = item["question"]
        ground_truth = item["answer"]

        print(f"[{i+1}/{total}] Processing: {question[:60]}...")

        retrieved = hybrid_retrieve(question)
        prompt    = build_cot_prompt(question, retrieved)
        answer    = generate(prompt)

        results.append({
            "question":     question,
            "answer":       answer,
            "contexts":     retrieved,
            "ground_truth": ground_truth,
        })

        time.sleep(1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. {total} results saved to {output_path}")

# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nMode:")
    print("  1 — Interactive (ask one question)")
    print("  2 — Batch (run all test questions)")
    mode = input("Choose mode [1/2]: ").strip()

    if mode == "2":
        output_file = "results_c9.json"
        run_batch("test_dataset.json", output_file)
    else:
        while True:
            q = input("\nAsk a question (or 'quit'): ").strip()
            if q.lower() == "quit":
                break
            print(f"\nFull Answer:\n{ask(q)}")