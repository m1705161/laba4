"""Стадия 3 — Оценка retrieval: recall@k и MRR по золотому набору gold.csv.

Формат gold.csv (заголовки обязательны):
    question,doc_id
    "что такое lakehouse?",1
    ...

Запуск (внутри python-контейнера):
  docker compose exec py python /app/03-вопрос-ответ-ui/evaluate.py --chunk-size 200 --gold /app/03-вопрос-ответ-ui/eval/gold.csv
"""
import argparse
import csv

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://qdrant:6333"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--gold", default="/app/03-вопрос-ответ-ui/eval/gold.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--recall-at", type=int, default=3)
    args = ap.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    qd = QdrantClient(url=QDRANT_URL)

    with open(args.gold, encoding="utf-8") as f:
        gold = list(csv.DictReader(f))

    hits, mrr = 0, 0.0
    for g in gold:
        qv = model.encode(g["question"]).tolist()
        res = qd.query_points(collection_name=f"chunks_{args.chunk_size}",
                              query=qv, limit=args.k).points
        got = [h.payload["doc_id"] for h in res]
        want = int(g["doc_id"])
        if want in got[:args.recall_at]:
            hits += 1
        rank = next((i + 1 for i, d in enumerate(got) if d == want), None)
        if rank:
            mrr += 1.0 / rank
        print(f"Q: {g['question'][:52]:<52} gold={want} top-{args.k}={got}")

    n = len(gold)
    print(f"\nchunk_size={args.chunk_size}: "
          f"recall@{args.recall_at} = {hits}/{n} = {hits / n:.2f}, "
          f"MRR = {mrr / n:.3f}")


if __name__ == "__main__":
    main()
