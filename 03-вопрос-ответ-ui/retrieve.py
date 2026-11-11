"""Стадия 3 — Retriever: вопрос -> top-k чанков из Qdrant.

Запуск (внутри python-контейнера):
  docker compose exec py python /app/03-вопрос-ответ-ui/retrieve.py --chunk-size 500 "что такое lakehouse?"
"""
import argparse

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://qdrant:6333"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("question")
    args = ap.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    qd = QdrantClient(url=QDRANT_URL)
    qv = model.encode(args.question).tolist()

    hits = qd.query_points(collection_name=f"chunks_{args.chunk_size}",
                           query=qv, limit=args.k).points

    print(f"Вопрос: {args.question}\n")
    for h in hits:
        p = h.payload
        print(f"doc_id={p['doc_id']}  chunk_id={h.id}  sim={h.score:.3f}")
        print("  " + p["chunk_text"][:180].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
