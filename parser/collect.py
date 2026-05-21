"""
Оркестратор сбора отзывов с Kaspi.kz по списку URL из urls.yaml.

Логика:
- Идёт по категориям из urls.yaml
- Для каждой категории парсит товары по очереди
- Применяет cap (макс отзывов с одного товара) и target (цель по категории)
- Фильтры: длина 20–150 слов, дедупликация по id
- Сохраняет в JSONL построчно (стримингом)
- Поддерживает checkpoint: при перезапуске пропускает уже обработанные URL

Запуск:
    python parser/collect.py --urls parser/urls.yaml --out data/raw/reviews.jsonl

Опции:
    --jitter-min 2 --jitter-max 6   задержка между товарами (секунды, антибот)
    --max-reviews 300               максимум отзывов парсить с одного товара (до cap)
    --resume                        продолжить с чекпоинта (по умолчанию включено)
    --reset                         удалить чекпоинт и начать с нуля
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import yaml

from scraper import scrape_product_reviews


# ───────────────────────────── Утилиты ─────────────────────────────


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def filter_review(r: dict, min_words: int, max_words: int) -> bool:
    """True если отзыв проходит фильтры качества."""
    text = (r.get("text") or "").strip()
    if not text:
        return False
    wc = word_count(text)
    if wc < min_words or wc > max_words:
        return False
    return True


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"processed_urls": [], "counts": {}, "seen_ids": []}


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    # Срезаем комментарии и хвостовой мусор
    if u.startswith("#") or not u.startswith("http"):
        return ""
    return u


# ───────────────────────────── Основной цикл ─────────────────────────────


def collect(
    urls_path: Path,
    out_path: Path,
    checkpoint_path: Path,
    jitter_min: float,
    jitter_max: float,
    max_reviews_per_product: int,
    reset: bool,
) -> None:
    cfg_raw = yaml.safe_load(urls_path.read_text(encoding="utf-8"))
    cfg = cfg_raw.get("config", {})
    per_product_cap = int(cfg.get("per_product_cap", 80))
    per_category_target = int(cfg.get("per_category_target", 500))
    min_words = int(cfg.get("min_review_words", 20))
    max_words = int(cfg.get("max_review_words", 150))

    categories_block = cfg_raw.get("categories", {})

    if reset and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[ckpt] Чекпоинт удалён: {checkpoint_path}")

    state = load_checkpoint(checkpoint_path)
    processed_urls = set(state.get("processed_urls", []))
    seen_ids = set(state.get("seen_ids", []))
    counts: dict[str, int] = state.get("counts", {})

    print("=" * 70)
    print(f"Конфиг:")
    print(f"  per_product_cap     = {per_product_cap}")
    print(f"  per_category_target = {per_category_target}")
    print(f"  min_review_words    = {min_words}")
    print(f"  max_review_words    = {max_words}")
    print(f"  jitter              = {jitter_min}–{jitter_max}s между товарами")
    print(f"Категорий в urls.yaml: {len(categories_block)}")
    print(f"Уже обработано URL: {len(processed_urls)}")
    print(f"Уже собрано отзывов в чекпоинте: {sum(counts.values())}")
    print("=" * 70)

    total_added_session = 0
    summary: dict[str, dict] = {}

    for category, raw_urls in categories_block.items():
        if not isinstance(raw_urls, list):
            print(f"[{category}] нет URL в yaml, пропускаю")
            continue

        urls = [normalize_url(u) for u in raw_urls]
        urls = [u for u in urls if u]

        collected = counts.get(category, 0)
        summary[category] = {"collected_before": collected, "added": 0, "urls_total": len(urls)}

        if not urls:
            print(f"[{category}] URL пустой список — пропуск")
            continue

        if collected >= per_category_target:
            print(f"[{category}] квота {per_category_target} уже выполнена, пропуск ({collected}/{per_category_target})")
            continue

        print(f"\n[{category}] target={per_category_target}, уже собрано={collected}, URL в списке={len(urls)}")

        for url in urls:
            if collected >= per_category_target:
                print(f"[{category}] ✅ квота достигнута: {collected}/{per_category_target}")
                break

            if url in processed_urls:
                print(f"[{category}] skip (уже обработан): {url[:70]}")
                continue

            remaining = per_category_target - collected
            print(f"[{category}] парсю {url[:70]}... (нужно ещё {remaining})")

            try:
                title, reviews = scrape_product_reviews(url, max_reviews=max_reviews_per_product)
            except Exception as e:
                print(f"[{category}] ❌ ошибка парсинга: {e}")
                # URL не помечаем processed — можно будет переторкать
                time.sleep(random.uniform(jitter_min, jitter_max))
                continue

            # Фильтр по длине ДО обрезки (ведь парсер скачивает до 300 отзывов)
            before_filter = len(reviews)
            reviews = [r for r in reviews if filter_review(r, min_words, max_words)]

            # Cap на один товар (берём только отфильтрованные)
            reviews = reviews[:per_product_cap]

            # Дедуп по id (внутри сессии и между запусками)
            new_reviews = []
            for r in reviews:
                rid = r["id"]
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                r["category"] = category
                r["product_title"] = title
                new_reviews.append(r)

            # Обрезка под остаток квоты
            new_reviews = new_reviews[:remaining]

            # Сохраняем стримом
            append_jsonl(out_path, new_reviews)

            added = len(new_reviews)
            collected += added
            total_added_session += added
            counts[category] = collected
            processed_urls.add(url)
            summary[category]["added"] += added

            print(
                f"[{category}]   title='{title[:50]}' "
                f"raw={before_filter} -> filt={len(reviews)} -> new={added} "
                f"| итого {collected}/{per_category_target}"
            )

            # Persist checkpoint
            state["processed_urls"] = sorted(processed_urls)
            state["counts"] = counts
            state["seen_ids"] = sorted(seen_ids)
            save_checkpoint(checkpoint_path, state)

            # Антибот jitter
            delay = random.uniform(jitter_min, jitter_max)
            time.sleep(delay)

        # Итог категории
        if collected < per_category_target:
            print(
                f"[{category}] ⚠️ недобор: {collected}/{per_category_target} "
                f"— добавь ещё URL в urls.yaml и перезапусти"
            )
        else:
            print(f"[{category}] ✅ готово: {collected}/{per_category_target}")

    # Финальный отчёт
    print("\n" + "=" * 70)
    print("ИТОГО ЗА СЕССИЮ")
    print("=" * 70)
    grand_total = 0
    for cat, info in summary.items():
        total = counts.get(cat, 0)
        grand_total += total
        status = "✅" if total >= per_category_target else "⚠️"
        print(
            f"  {status} {cat:20s} "
            f"всего={total:4d}/{per_category_target}  "
            f"(добавлено в сессии: {info['added']})"
        )
    print(f"\n  В сессии добавлено отзывов: {total_added_session}")
    print(f"  Всего в датасете:           {grand_total}")
    print(f"  Файл данных:                {out_path}")
    print(f"  Чекпоинт:                   {checkpoint_path}")


def main():
    ap = argparse.ArgumentParser(description="Сбор отзывов Kaspi.kz по urls.yaml")
    ap.add_argument("--urls", type=Path, default=Path("parser/urls.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/reviews.jsonl"))
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/raw/_checkpoint.json"),
    )
    ap.add_argument("--jitter-min", type=float, default=2.0)
    ap.add_argument("--jitter-max", type=float, default=6.0)
    ap.add_argument(
        "--max-reviews",
        type=int,
        default=300,
        help="Сколько отзывов максимум парсить с одного товара (до cap из yaml)",
    )
    ap.add_argument("--reset", action="store_true", help="Стереть чекпоинт и начать с нуля")
    args = ap.parse_args()

    if not args.urls.exists():
        print(f"❌ urls.yaml не найден: {args.urls}", file=sys.stderr)
        sys.exit(1)

    collect(
        urls_path=args.urls,
        out_path=args.out,
        checkpoint_path=args.checkpoint,
        jitter_min=args.jitter_min,
        jitter_max=args.jitter_max,
        max_reviews_per_product=args.max_reviews,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
