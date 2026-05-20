# GAN-Reviews: Сравнительный анализ генеративных моделей для текстовых отзывов

Финальный проект по дисциплине **Generative Adversarial Networks** (магистратура).

> Если ты AI-ассистент на новом ПК — **сначала прочитай [HANDOFF.md](HANDOFF.md)**.

---

## Идея проекта

Исследовать границы применимости GAN для генерации текстовых данных на примере
отзывов о товарах с Kaspi.kz.

**Главный вопрос:** где проходит граница применимости GAN — на уровне
эмбеддингов (непрерывное пространство) или на уровне токенов (дискретное)?

**Подход:**

| Уровень | Модель | Чем сравниваем |
|---------|--------|----------------|
| Эмбеддинги (768-dim XLM-R) | **WGAN-GP** | Основная модель |
| Эмбеддинги | **VAE** | Бейзлайн другого семейства |
| Токены (текст) | **Text-GAN** (Gumbel-Softmax) | Демонстрация ограничений |

---

## Структура

```
gan-reviews/
├── HANDOFF.md                    ← для AI на новом ПК
├── README.md
├── docker-compose.yml
├── docker/                       Dockerfile для parser и trainer
├── parser/                       Сбор отзывов с Kaspi.kz
├── embeddings/                   Извлечение XLM-R эмбеддингов
├── models/                       WGAN-GP, VAE, Text-GAN
├── metrics/                      MMD, Frechet Distance
├── notebooks/                    Точки запуска (Colab-совместимые)
├── data/                         (gitignored) сырые и обработанные данные
├── checkpoints/                  (gitignored) сохранённые модели
├── results/                      графики и таблицы метрик
└── report/                       отчёт + презентация
```

Детали — в [HANDOFF.md](HANDOFF.md).

---

## Датасет

5000 отзывов с Kaspi.kz, **10 категорий × 500 отзывов**:

1. Смартфоны и аксессуары
2. Ноутбуки
3. Крупная бытовая техника
4. Мелкая бытовая техника
5. Одежда и обувь
6. Косметика
7. Продукты
8. Детские товары
9. Спорт
10. Мебель

Сбор стратифицированный, по 1–5 товаров на категорию. Фильтры: 5–100 слов,
дедуп по тексту.

---

## Модели

### 1. WGAN-GP на эмбеддингах

- **Generator:** MLP, (z=128) + label_emb → 768-dim вектор
- **Critic:** MLP, 768 + label_emb → scalar
- **Loss:** Wasserstein + Gradient Penalty (λ=10)
- **Trick:** LayerNorm в critic (BatchNorm нарушает Lipschitz)

### 2. Conditional VAE

- **Encoder/Decoder:** MLP
- **Latent:** 64-dim
- **Loss:** ELBO = reconstruction (MSE) + β·KL

### 3. Text-GAN (TODO)

- **Generator:** маленький Transformer decoder + Gumbel-Softmax
- **Discriminator:** LSTM или CNN
- **Loss:** adversarial
- **Ожидание:** будет генерировать кашу — это часть исследования

---

## Метрики

| Метрика | Что показывает |
|---------|----------------|
| **MMD** (Maximum Mean Discrepancy) | Близость распределений в пространстве признаков |
| **Frechet Distance** | Близость через моменты (μ, Σ) — аналог FID |
| **Per-class MMD/FD** | Где модель работает лучше/хуже |
| **t-SNE / UMAP** | Визуальное смешивание реальных и сгенерированных точек |

---

## Запуск через Docker

Ничего не ставится в систему — всё работает в контейнерах.

### Парсер Kaspi (локально)

1. Заполнить [parser/urls.yaml](parser/urls.yaml) ссылками на товары по 10 категориям.
2. Собрать образ (первый раз или после правок Dockerfile):
   ```bash
   docker compose build parser
   ```
3. Запустить сбор:
   ```bash
   docker compose run --rm parser
   ```
   Результат: `data/raw/reviews.jsonl`, чекпоинт: `data/raw/_checkpoint.json`.

   При обрыве — просто запусти ту же команду, продолжит с чекпоинта.

   Сбросить и начать с нуля:
   ```bash
   docker compose run --rm parser python parser/collect.py --reset
   ```

### Trainer (опционально — для локального Jupyter)

```bash
docker compose build trainer
docker compose up trainer
```

Открыть http://localhost:8888 (без пароля).

### Обучение моделей в Colab (рекомендуется)

1. Запушить репо на GitHub
2. В Colab: **File → Open notebook → GitHub** → выбрать ноутбук
3. Запустить ячейки — первая клонирует репо, ставит зависимости
4. После обучения: **File → Save a copy in GitHub** — outputs попадут в git

---

## Workflow

```
[parser/urls.yaml]
        │ ручное заполнение ~200–400 URL
        ▼
[docker compose run --rm parser]
        │
        ▼
[data/raw/reviews.jsonl]   ← 5000 отзывов
        │
        ▼
[notebooks/01_explore_data.ipynb]   ← EDA: длины, языки, рейтинги
        │
        ▼
[notebooks/02_extract_embeddings.ipynb]   ← XLM-R на T4
        │
        ▼
[data/embeddings/{X_cls,X_mean,labels,ratings}.npy]
        │
        ├──► [notebooks/03_train_wgan_gp.ipynb]   → checkpoints/wgan_gp.pth
        ├──► [notebooks/04_train_vae.ipynb]        → checkpoints/vae.pth
        └──► [notebooks/06_train_text_gan.ipynb]   → checkpoints/text_gan.pth
                │
                ▼
        [notebooks/05_compare_models.ipynb]   ← MMD, FD, t-SNE
                │
                ▼
        [notebooks/07_final_demo.ipynb]   ← end-to-end для защиты
                │
                ▼
        [report/report.md → report.docx]
```

---

## Параметры обучения (defaults)

| Параметр | Значение | Где задано |
|----------|----------|-----------|
| z_dim (WGAN-GP) | 128 | `models/wgan_gp.py` |
| hidden | 512 | `models/wgan_gp.py`, `models/vae.py` |
| n_critic | 5 | `models/wgan_gp.py` |
| lambda_gp | 10.0 | `models/wgan_gp.py` |
| Adam betas | (0.5, 0.9) | `models/wgan_gp.py` |
| lr | 1e-4 (GAN) / 1e-3 (VAE) | `models/*.py` |
| batch_size | 64 | в ноутбуках |
| n_epochs | 200 (GAN) / 100 (VAE) | в ноутбуках |
| latent_dim (VAE) | 64 | `models/vae.py` |
| β (VAE) | 1.0 | `models/vae.py` |

---

## Результаты

Будут добавлены после Final-этапа.

---

## Лицензия и автор

Sultan Khassenov, магистратура AITU.
