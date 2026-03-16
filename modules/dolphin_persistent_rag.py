#!/usr/bin/env python3
"""
Dolphin with persistent semantic memory using Chroma + sentence-transformers + Ollama
Minimal but production-grade version — stores conversation chunks, recalls semantically
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings
from ollama import Client
from sentence_transformers import SentenceTransformer

# ─── Config ────────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
MODEL = "dolphin-mistral-24b-venice-edition"          # ← change if your tag is different

MEMORY_PATH = Path("./dolphin_memory")
MEMORY_PATH.mkdir(exist_ok=True)

# ─── Components ────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")     # fast & decent quality

ollama = Client(host=OLLAMA_HOST)

chroma_client = chromadb.PersistentClient(
    path=str(MEMORY_PATH / "chroma_db"),
    settings=Settings(
        allow_reset=True,
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection(
    name="conversation_memory",
    metadata={"hnsw:space": "cosine"}
)

# ─── Core Functions ────────────────────────────────────────────────────────
def embed(text: str) -> List[float]:
    """Generate normalized embedding"""
    return embedder.encode(text, normalize_embeddings=True).tolist()


def remember(conversation_chunk: Dict[str, Any], tags: List[str] = None):
    """
    Store a conversation turn / summary / note in vector DB
    """
    text = json.dumps(conversation_chunk, ensure_ascii=False, indent=None)
    vector = embed(text)

    metadata = {
        "timestamp": conversation_chunk.get("timestamp", datetime.utcnow().isoformat()),
        "role": conversation_chunk.get("role", "unknown"),
        "tags": tags or []
    }

    collection.add(
        documents=[text],
        embeddings=[vector],
        metadatas=[metadata],
        ids=[f"mem_{hash(text)}"]
    )

    print(f"[Memory stored] {len(text)} chars")


def recall(query: str, n_results: int = 6, min_score: float = 0.22) -> List[Dict]:
    """
    Semantic search over stored memories
    Returns list of parsed memory dicts sorted by relevance (highest first)
    """
    if not query.strip():
        return []

    vector = embed(query)
    results = collection.query(
        query_embeddings=[vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    memories = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if dist <= (1 - min_score):  # cosine distance → lower is better
            try:
                parsed = json.loads(doc)
                parsed["relevance"] = 1 - dist
                memories.append(parsed)
            except json.JSONDecodeError:
                pass  # skip corrupt entries

    return sorted(memories, key=lambda x: x["relevance"], reverse=True)


# ─── Chat with Memory ──────────────────────────────────────────────────────
def chat_with_memory(user_input: str, max_context_memories: int = 4) -> str:
    """
    Main chat function with semantic memory injection
    """
    # 1. Recall relevant memories
    memories = recall(user_input)
    memory_context = ""
    if memories:
        memory_context = "\nRelevant past memories (most relevant first):\n" + "\n".join(
            f"[{m.get('timestamp','?')}] {m.get('role','?')}: {m.get('content','')[:300]}..."
            for m in memories[:max_context_memories]
        )

    # 2. Build system prompt with memory
    system_prompt = f"""You are Dolphin — uncensored, persistent, truth-seeking.
You have long-term memory. Use the provided memories when relevant — they are real past interactions.
Do not repeat memory content verbatim unless necessary.

{memory_context.strip()}

Answer naturally, stay in character, be concise and direct."""

    # 3. Call Ollama
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_input}
        ]
    )

    answer = response["message"]["content"].strip()

    # 4. Remember this exchange (you can make this conditional later)
    now = datetime.utcnow().isoformat()
    remember({
        "timestamp": now,
        "role": "assistant",
        "content": answer,
        "user_input": user_input
    }, tags=["conversation", "important"])

    return answer


# ─── CLI Loop ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Dolphin with persistent semantic RAG memory")
    print("Type 'quit', 'exit' or 'q' to stop")
    print("-" * 50)

    while True:
        try:
            user_msg = input("\nYou: ").strip()
            if user_msg.lower() in ("quit", "exit", "q", ""):
                print("\nGoodbye.")
                break

            if not user_msg:
                continue

            reply = chat_with_memory(user_msg)
            print(f"Dolphin: {reply}\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye.")
            break
        except Exception as e:
            print(f"[Error] {str(e)}")
