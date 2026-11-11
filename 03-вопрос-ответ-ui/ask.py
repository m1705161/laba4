"""Стадия 3 — Генерация: вопрос -> ответ через Ollama, с RAG и без (сравнение).

Перед первым запуском скачайте модель (один раз):
  docker compose exec ollama ollama pull qwen2.5:1.5b

Запуск (внутри python-контейнера):
  docker compose exec py python /app/03-вопрос-ответ-ui/ask.py --chunk-size 500 --model qwen2.5:1.5b "что такое lakehouse?"
  docker compose exec py python /app/03-вопрос-ответ-ui/ask.py --chunk-size 500 --model qwen2.5:1.5b --no-rag "что такое lakehouse?"
"""
import argparse

import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://qdrant:6333"
OLLAMA_URL = "http://ollama:11434"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def chat(model, prompt):
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "stream": False,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-rag", action="store_true",
                    help="спросить LLM БЕЗ контекста (чистая модель)")
    ap.add_argument("question")
    args = ap.parse_args()

    if args.no_rag:
        print(f"=== БЕЗ RAG (модель {args.model}, без контекста) ===\n")
        print(chat(args.model, args.question))
        return

    model = SentenceTransformer(MODEL_NAME)
    qd = QdrantClient(url=QDRANT_URL)
    qv = model.encode(args.question).tolist()
    hits = qd.query_points(collection_name=f"chunks_{args.chunk_size}",
                           query=qv, limit=args.k).points
    context = "\n".join(h.payload["chunk_text"] for h in hits)

    prompt = (
        "Отвечай ТОЛЬКО по контексту ниже. Если в контексте нет ответа — "
        "напиши, что не знаешь.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {args.question}"
    )
    print(f"=== С RAG (модель {args.model}, chunks_{args.chunk_size}) ===\n")
    print(chat(args.model, prompt))
    print("\n--- найденные чанки ---")
    for h in hits:
        print(f"  doc_id={h.payload['doc_id']}  sim={h.score:.3f}")


if __name__ == "__main__":
    main()
