"""
Извлечение эмбеддингов отзывов через XLM-RoBERTa.

Читает data/raw/reviews.jsonl, для каждого отзыва получает 768-dim вектор
[CLS]-токена + mean-pooling, сохраняет:
  data/embeddings/X.npy           (N, 768) — векторы
  data/embeddings/labels.npy      (N,) — category_id (0..9)
  data/embeddings/ratings.npy     (N,) — рейтинг 1..5
  data/embeddings/meta.json       — отображение category_id <-> name + статистика

Запуск:
    python embeddings/extract.py \
        --input data/raw/reviews.jsonl \
        --out data/embeddings/

Опции:
    --model xlm-roberta-base
    --pooling cls | mean | both     (default: both — concat не делает,
                                                  сохраняет два варианта)
    --batch-size 32
    --max-len 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class ReviewsDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling по non-padding токенам."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def extract_embeddings(
    reviews: list[dict],
    model_name: str = "xlm-roberta-base",
    pooling: str = "both",
    batch_size: int = 32,
    max_len: int = 128,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Возвращает словарь массивов:
      'cls':  (N, 768) — эмбеддинг [CLS]
      'mean': (N, 768) — mean-pooling
      'labels': (N,) — category_id
      'ratings': (N,) — рейтинги
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device}, model={model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    texts = [r["text"] for r in reviews]
    categories = sorted({r["category"] for r in reviews})
    cat_to_id = {c: i for i, c in enumerate(categories)}
    labels = np.array([cat_to_id[r["category"]] for r in reviews], dtype=np.int64)
    ratings = np.array([int(r.get("rating", 0)) for r in reviews], dtype=np.int64)

    dataset = ReviewsDataset(texts, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    cls_chunks, mean_chunks = [], []
    for batch in tqdm(loader, desc="embedding"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=mask)
        last = out.last_hidden_state  # (B, T, 768)
        cls_chunks.append(last[:, 0, :].cpu().numpy())
        mean_chunks.append(mean_pool(last, mask).cpu().numpy())

    cls_arr = np.concatenate(cls_chunks, axis=0)
    mean_arr = np.concatenate(mean_chunks, axis=0)

    return {
        "cls": cls_arr,
        "mean": mean_arr,
        "labels": labels,
        "ratings": ratings,
        "cat_to_id": cat_to_id,
        "id_to_cat": {i: c for c, i in cat_to_id.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/raw/reviews.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/embeddings"))
    ap.add_argument("--model", type=str, default="xlm-roberta-base")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[extract] reading {args.input}")
    reviews = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    print(f"[extract] {len(reviews)} reviews")

    result = extract_embeddings(
        reviews,
        model_name=args.model,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )

    np.save(args.out / "X_cls.npy", result["cls"])
    np.save(args.out / "X_mean.npy", result["mean"])
    np.save(args.out / "labels.npy", result["labels"])
    np.save(args.out / "ratings.npy", result["ratings"])

    meta = {
        "model": args.model,
        "max_len": args.max_len,
        "n_reviews": len(reviews),
        "emb_dim": int(result["cls"].shape[1]),
        "cat_to_id": result["cat_to_id"],
        "id_to_cat": {str(k): v for k, v in result["id_to_cat"].items()},
        "n_per_cat": {
            cat: int(np.sum(result["labels"] == idx))
            for cat, idx in result["cat_to_id"].items()
        },
    }
    (args.out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[extract] saved to {args.out}")
    print(f"  X_cls.npy:  {result['cls'].shape}")
    print(f"  X_mean.npy: {result['mean'].shape}")
    print(f"  labels.npy: {result['labels'].shape}")
    print(f"  meta.json:  {meta['n_per_cat']}")


if __name__ == "__main__":
    main()
