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
