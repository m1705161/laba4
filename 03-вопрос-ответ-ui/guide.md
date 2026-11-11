# Стадия 3 — Вопрос → ответ в UI (~90 мин)

> Финал: превращаем вопрос пользователя в ответ нейросети, а под капот подставляем
> найденные чанки. Здесь же сравниваем **без RAG / с RAG** и меряем качество поиска.

## Что мы делаем (простыми словами)

До сих пор мы только **искали** (retrieval). Теперь подключаем **генерацию** (generation):

```
вопрос → эмбеддинг → top-k чанков (Qdrant) → контекст → LLM (Ollama) → ответ
```

Главная идея RAG — сравнить **два режима**:

- **Без RAG** — спрашиваем модель напрямую, она отвечает «из головы» (из того, что
  запомнила при обучении). На твоих специфических документах она будет **галлюцинировать**
  (выдумывать) или отвечать мимо.
- **С RAG** — сначала ищем нужные чанки в твоих документах, кладём их в промпт, и модель
  отвечает **по документам**.

## Шаг 1. Скачать LLM-модель в Ollama (один раз)

```bash
docker compose exec ollama ollama pull qwen2.5:0.5b
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama pull qwen2.5:3b
```

`ollama pull` качает модель в контейнер `ollama`. Выбери модель по памяти:

| Модель | ~память модели | Куда влезет |
|---|---|---|
| `qwen2.5:0.5b` | ~0.6 ГБ | 8 ГБ с запасом |
| `qwen2.5:1.5b` | ~1.5 ГБ | 8 ГБ — дефолт |
| `qwen2.5:3b` | ~3 ГБ | 16 ГБ; на 8 ГБ впритык |

Проверка, что Ollama жива: открой http://localhost:11434 (страница отвечает).

## Шаг 1.5. Создать скрипты (retrieve, ask, evaluate, app)

Скопируй четыре блока целиком — каждый создаст свой файл в `03-вопрос-ответ-ui/`.
Править в них ничего не нужно.

```bash
cat > 03-вопрос-ответ-ui/retrieve.py << 'EOF'
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
EOF
```

```bash
cat > 03-вопрос-ответ-ui/ask.py << 'EOF'
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
EOF
```

```bash
cat > 03-вопрос-ответ-ui/evaluate.py << 'EOF'
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
EOF
```

```bash
cat > 03-вопрос-ответ-ui/app.py << 'EOF'
"""Стадия 3 — Streamlit UI: вопрос -> топ-чанки + ответ (Ollama).

Запуск (из каталога лабы, держит терминал):
  docker compose exec py streamlit run /app/03-вопрос-ответ-ui/app.py \
      --server.address 0.0.0.0 --server.port 8501
Открыть: http://localhost:8501
"""
import requests
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://qdrant:6333"
OLLAMA_URL = "http://ollama:11434"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@st.cache_resource
def load_embedder():
    return SentenceTransformer(MODEL_NAME)


def retrieve(question, chunk_size, k):
    emb = load_embedder().encode(question).tolist()
    qd = QdrantClient(url=QDRANT_URL)
    return qd.query_points(collection_name=f"chunks_{chunk_size}",
                           query=emb, limit=k).points


def chat(model, prompt):
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "stream": False,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


st.title("RAG: вопрос → ответ по своим документам")

chunk_size = st.selectbox("Коллекция (размер чанка)", [200, 500, 1000], index=1)
model = st.selectbox("LLM", ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"], index=1)
k = st.slider("top-k", 1, 10, 3)
use_rag = st.checkbox("Использовать RAG (подставлять контекст)", value=True)
question = st.text_input("Вопрос", placeholder="что такое lakehouse?")

if st.button("Спросить") and question:
    if use_rag:
        hits = retrieve(question, chunk_size, k)
        context = "\n".join(h.payload["chunk_text"] for h in hits)
        prompt = f"Отвечай ТОЛЬКО по контексту:\n{context}\n\nВопрос: {question}"
        with st.spinner("Генерация..."):
            answer = chat(model, prompt)
        st.markdown("### Ответ (с RAG)")
        st.write(answer)
        st.markdown("### Найденные чанки")
        for h in hits:
            st.markdown(f"**doc_id={h.payload['doc_id']}** · sim={h.score:.3f}")
            st.text(h.payload["chunk_text"][:300])
    else:
        with st.spinner("Генерация..."):
            answer = chat(model, question)
        st.markdown("### Ответ (без RAG)")
        st.write(answer)
EOF
```

## Шаг 2. Проверить поиск (retriever) из консоли

```bash
docker compose exec py python /app/03-вопрос-ответ-ui/retrieve.py --chunk-size 500 "что такое lakehouse?"
```

`retrieve.py`: берёт вопрос → считает его эмбеддинг той же MiniLM → ищет в Qdrant топ
похожих чанков → печатает их с близостью `sim` и `doc_id`.

Что увидишь: список чанков с `doc_id=...`, `chunk_id=...`, `sim=0.xxx` и куском текста.

## Шаг 3. ГЛАВНОЕ — без RAG vs с RAG

Возьми вопрос по своей сфере, на который модель **не знает ответа из памяти** (что-то
специфичное из твоих документов).

Сначала **без RAG** (чистая модель, флаг `--no-rag`):

```bash
docker compose exec py python /app/03-вопрос-ответ-ui/ask.py --chunk-size 500 --model qwen2.5:1.5b --no-rag "твой вопрос"
```

Потом **с RAG** (тот же вопрос, флаг убран):

```bash
docker compose exec py python /app/03-вопрос-ответ-ui/ask.py --chunk-size 500 --model qwen2.5:1.5b "твой вопрос"
```

Что делает `ask.py` в режиме «с RAG»:
1. Считает эмбеддинг вопроса.
2. Ищет top-k чанков в Qdrant.
3. Склеивает их в «контекст» и кладёт в промпт: «Отвечай ТОЛЬКО по контексту…».
4. Шлёт промпт в Ollama (`/api/chat`) и печатает ответ + найденные чанки.

Сравни два ответа. **Ожидаемый результат:** без RAG — бред/галлюцинации или общая фраза
мимо; с RAG — ответ именно из твоих документов (иногда почти дословно). Запиши оба
ответа в отчёт `отчёт-исследование.md`.

Пример реального прогона (сфера «Химическая технология», вопрос «На какие пять групп
делят процессы химической технологии по кинетическим закономерностям?»):

- **Без RAG:** «…Средние-средние процессы… технические процессы… химическая
  термоядеризация…» и пункты 8, 9, 10 вместо пяти — полная галлюцинация.
- **С RAG:** «1. Гидромеханические, 2. Тепловые, 3. Массообменные, 4. Химические,
  5. Механические» — ровно пять групп из документа.

## Шаг 4. Запустить UI (Streamlit)

```bash
docker compose exec py streamlit run /app/03-вопрос-ответ-ui/app.py \
    --server.address 0.0.0.0 --server.port 8501
```

Открой http://localhost:8501 — там:
- меню коллекции (200/500/1000),
- меню LLM (0.5b/1.5b/3b),
- число top-k,
- поле вопроса,
- чекбокс «с RAG / без RAG»,
- ответ + найденные чанки.

Терминал с этой командой **не закрывай** (он держит сервер). Остановить — `Ctrl+C`.

## Шаг 5. Замерить качество поиска (recall@3 / MRR)

Сначала напиши **золотой набор** — файл `gold.csv` в папке твоей сферы. Формат:

```csv
question,doc_id
"что такое добыча нефти?",1
"как устроена доменная печь?",2
```

Каждая строка = вопрос и номер **правильного** документа (`doc_id` из имени файла).
Достаточно 5–8 вопросов — по одному-два на документ. Это твоя «эталонная правда».

Затем запусти оценку для каждой коллекции:

```bash
docker compose exec py python /app/03-вопрос-ответ-ui/evaluate.py --chunk-size 200 --gold /app/documents/cs/gold.csv
docker compose exec py python /app/03-вопрос-ответ-ui/evaluate.py --chunk-size 500 --gold /app/documents/cs/gold.csv
docker compose exec py python /app/03-вопрос-ответ-ui/evaluate.py --chunk-size 1000 --gold /app/documents/cs/gold.csv
```

> `/app/documents/cs/gold.csv` — пример для сферы cs; подставь путь к своему `gold.csv`.

Что делает `evaluate.py`: для каждого вопроса ищет топ-k и проверяет, попал ли правильный
`doc_id` в результат. Считает:
- **recall@3** — доля вопросов, где правильный документ попал в топ-3 (от 0 до 1);
- **MRR** — среднее «1/ранг первого правильного чанка» (ближе к 1 = выше).

Что увидишь: по каждому вопросу строку `Q: ... gold=N top-5=[...]` и в конце
`recall@3 = ..., MRR = ...`.

## Чек-пойнт (проверь, что получилось)

1. В UI вопрос даёт ответ по твоим документам + видны найденные чанки.
2. «Без RAG» и «с RAG» дают РАЗНЫЕ ответы (с RAG — по документам).
3. recall@3 / MRR посчитаны для всех трёх коллекций.
4. Отчёт `отчёт-исследование.md` заполнен (таблицы + вывод).

## Если что-то пошло не так / как откатиться

**Поменять модель/вопрос** — просто перезапусти команду. Данные это не меняет.

**Остановить Streamlit** — `Ctrl+C` в терминале, где он запущен.

**Пересчитать оценку** — поправь `gold.csv` и заново запусти `evaluate.py`.

**Полный сброс всей лабы** — как в стадиях 1–2: `docker compose down -v`, затем
`up -d` + `pip install` + `ollama pull`.
