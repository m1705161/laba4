"""Стадия 1 — Загрузка документов в MinIO + реестр manifest.json.

Читает .md из локальной папки (./documents/<сфера>), заливает каждый файл в MinIO
(bucket `rag`, префикс `documents/`) и пишет реестр manifest.json.

Запуск (внутри python-контейнера):
  docker compose exec py python /app/01-загрузка-реестр/loader.py --folder documents/cs
"""
import argparse
import json
import os
import time

from minio import Minio

MINIO_HOST = "minio:9000"
ACCESS = "minioadmin"
SECRET = "minioadmin"
BUCKET = "rag"
PREFIX = "documents/"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", required=True,
                    help="папка с документами, напр. documents/cs")
    args = ap.parse_args()

    mc = Minio(MINIO_HOST, access_key=ACCESS, secret_key=SECRET, secure=False)
    if not mc.bucket_exists(BUCKET):
        mc.make_bucket(BUCKET)
        print(f"[load] создан bucket s3://{BUCKET}")

    manifest = []
    for name in sorted(os.listdir(args.folder)):
        if not name.endswith(".md"):
            continue
        # "01-lakehouse.md" -> doc_id = 1, title = "lakehouse"
        head = name.split("-", 1)[0]
        if not head.isdigit():
            continue          # пропускаем README.md и прочие файлы не вида NN-название.md
        doc_id = int(head)
        title = name.split("-", 1)[1].removesuffix(".md").replace("-", " ")
        obj = f"{PREFIX}{name}"
        mc.fput_object(BUCKET, obj, os.path.join(args.folder, name))
        manifest.append({
            "doc_id": doc_id,
            "title": title,
            "path": obj,
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"[load] {name} -> s3://{BUCKET}/{obj}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[load] готово: {len(manifest)} документов, реестр {out}")


if __name__ == "__main__":
    main()
