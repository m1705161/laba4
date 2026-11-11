"""Стадия 2 — Чанкинг + эмбеддинги + запись в Qdrant.

Читает документы из MinIO (bucket `rag`, префикс `documents/`), режет каждый на чанки
(окно `--chunk-size` + `--overlap`), считает эмбеддинги MiniLM (384d) и пишет в коллекцию
Qdrant `chunks_<size>`. Запустите с разными --chunk-size, чтобы получить все три коллекции.

Запуск (внутри python-контейнера):
  docker compose exec py python /app/02-чанки-эмбеддинги-векторы/chunk_embed.py --chunk-size 200
  docker compose exec py python /app/02-чанки-эмбеддинги-векторы/chunk_embed.py --chunk-size 500
  docker compose exec py python /app/02-чанки-эмбеддинги-векторы/chunk_embed.py --chunk-size 1000
"""
import argparse

from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

MINIO_HOST = "minio:9000"
ACCESS = "minioadmin"
SECRET = "minioadmin"
BUCKET = "rag"
PREFIX = "documents/"
QDRANT_URL = "http://qdrant:6333"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384


def chunk(text, size, overlap):
    """Режем текст на чанки фиксированного окна с перекрытием."""
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--overlap", type=int, default=40)
    args = ap.parse_args()
    coll = f"chunks_{args.chunk_size}"

    model = SentenceTransformer(MODEL_NAME)          # первый запуск качает модель (~470 МБ)
    mc = Minio(MINIO_HOST, access_key=ACCESS, secret_key=SECRET, secure=False)
    qd = QdrantClient(url=QDRANT_URL)

    if qd.collection_exists(coll):
        qd.delete_collection(coll)
    qd.create_collection(
        coll,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )

    docs = [o.object_name for o in mc.list_objects(BUCKET, prefix=PREFIX, recursive=True)
            if o.object_name.endswith(".md")]
    if not docs:
        print("[chunk] в MinIO нет .md — сначала выполните стадию 1 (loader.py)")
        return

    points, chunk_id = [], 1
    for name in sorted(docs):
        doc_id = int(name.split("/")[-1].split("-")[0])
        text = mc.get_object(BUCKET, name).read().decode("utf-8")
        for i, piece in enumerate(chunk(text, args.chunk_size, args.overlap)):
            emb = model.encode(piece).tolist()
            points.append(PointStruct(
                id=chunk_id,
                vector=emb,
                payload={
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "chunk_text": piece,
                    "num_chars": len(piece),
                },
            ))
            chunk_id += 1

    qd.upsert(collection_name=coll, points=points)
    print(f"[chunk] {chunk_id - 1} чанков -> Qdrant collection '{coll}'")


if __name__ == "__main__":
    main()
