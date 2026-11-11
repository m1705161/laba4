# Стадия 2 — Чанки → эмбеддинги → векторы в Qdrant (Windows) (~90 мин)

> То же, что в `guide.md`, но команды — для **PowerShell на Windows**. Пути внутри
> контейнера (`/app/...`) не меняются.

## Что мы делаем (простыми словами)

Нейросеть не «понимает» текст как есть — ей нужны **числа**. Поэтому:

1. **Чанкинг** — режем документ на куски (чанки) подходящего размера.
2. **Эмбеддинг** — превращаем каждый чанк в вектор из 384 чисел. Похожие по смыслу
   тексты дают близкие векторы.
3. **Запись в Qdrant** — кладём векторы в векторную базу, где можно быстро искать
   «ближайших соседей».

Пайплайн в одну строку:

```
документ (MinIO) → чанки → эмбеддинг (MiniLM, 384 числа) → Qdrant (коллекция chunks_N)
```

## Понятия, которые надо знать

- **Чанк (chunk)** — кусок текста. Размер в символах: `chunk_size=500` ≈ куски по 500
  символов.
- **Overlap (перекрытие)** — сколько символов из предыдущего куска попадает в следующий.
  Нужно, чтобы не разрезать мысль ровно на границе. У нас `overlap=40`.
- **Эмбеддинг (embedding)** — вектор-«смысловая подпись» текста. Модель
  `paraphrase-multilingual-MiniLM-L12-v2` даёт 384 числа и понимает русский.
- **Qdrant** — векторная база. Данные лежат в **коллекциях** (collection). Элемент —
  **точка** (point) = вектор + **payload** (текст чанка и `doc_id`).
- **Косинусная близость** — мера «насколько векторы похожи», от −1 до 1. Ближе к 1 —
  ближе по смыслу.

## Шаг 1. Создать скрипт chunk_embed.py и прогнать чанкинг

Сначала создай скрипт — скопируй блок целиком (создаст файл
`02-чанки-эмбеддинги-векторы\chunk_embed.py`):

```powershell
[System.IO.File]::WriteAllText("$PWD\02-чанки-эмбеддинги-векторы\chunk_embed.py", @'
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
'@)
```

Теперь запускаем его для трёх размеров чанка:

```powershell
docker compose exec py python /app/02-чанки-эмбеддинги-векторы/chunk_embed.py --chunk-size 200
docker compose exec py python /app/02-чанки-эмбеддинги-векторы/chunk_embed.py --chunk-size 500
docker compose exec py python /app/02-чанки-эмбеддинги-векторы/chunk_embed.py --chunk-size 1000
```

Что делает `chunk_embed.py`:
1. Читает все `.md` из MinIO (бакет `rag`).
2. Режет каждый на чанки `--chunk-size` с `--overlap=40`.
3. Считает для каждого чанка эмбеддинг MiniLM (384 числа).
4. Создаёт коллекцию `chunks_<size>` и кладёт туда точки (вектор + payload: `chunk_text`,
   `doc_id`, `chunk_index`, `num_chars`).

Три запуска с разным `--chunk-size` → три коллекции: `chunks_200`, `chunks_500`,
`chunks_1000`. Это наш **варьируемый параметр** — в стадии 3 замерим, какой размер ищет
точнее.

Что увидишь: `[chunk] N чанков -> Qdrant collection 'chunks_200'` и т.д. Чем крупнее
чанк, тем меньше чанков.

> ⏱️ Первый запуск качает модель MiniLM (~470 МБ). Дальше берётся из кэша — быстро.

## Шаг 2. Проверить коллекции в Qdrant

Открой встроенный dashboard Qdrant: http://localhost:6333/dashboard

- Видны 3 коллекции: `chunks_200`, `chunks_500`, `chunks_1000`.
- У каждой — число точек (чанков) и размерность вектора **384**.

## Чек-пойнт (проверь, что получилось)

- Три коллекции существуют, в каждой — чанки.
- Логика верная: `chunk_size` больше → чанков меньше (200 → много, 1000 → мало).
- У чанка в payload есть `doc_id` (из какого документа), `chunk_text` (текст), `num_chars`.

## Почему размер чанка — это компромисс (главная мысль)

- **Мелкие (200):** точнее попадание в вопрос, меньше «воды», но мысль может быть
  разрезана пополам и контекст потерян.
- **Крупные (1000):** больше контекста для LLM, но в топ попадает много лишнего и
  косинусная близость «размазывается».
- Какой лучше — **не угадываем, а меряем** в стадии 3 (recall@3 / MRR).

> 💡 Qdrant ищет **приближённо** (ANN, индекс HNSW) — компромисс «скорость ↔ полнота».
> На маленьком корпусе разницы нет; на миллионах векторов HNSW даёт поиск за
> миллисекунды ценой доли процента recall.

## Если что-то пошло не так / как откатиться

**Переделать только стадию 2** — просто повтори Шаг 1: `chunk_embed.py` сам удаляет и
пересоздаёт коллекцию с тем же именем. Следов не остаётся.

**Удалить коллекции вручную** (начать стадию 2 с нуля):

```powershell
docker compose exec py python -c "from qdrant_client import QdrantClient; c=QdrantClient(url='http://qdrant:6333'); [c.delete_collection(n) for n in ['chunks_200','chunks_500','chunks_1000'] if c.collection_exists(n)]; print('collections deleted')"
```

> Документы в MinIO эта команда **не трогает** — они остаются, можно сразу заново
> прогнать `chunk_embed.py`.

**Полный сброс всей лабы** — как в конце стадии 1: `docker compose down -v`, затем
`up -d` + `pip install` + `ollama pull`.
