"""
Генератор скелетов Jupyter-ноутбуков для проекта.

Запускается один раз для создания шаблонов:
    python notebooks/_generate_skeletons.py

После этого ноутбуки наполняются вручную/AI/в Colab.
"""
import json
from pathlib import Path


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def save_nb(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"created {path}")


# ────────────────────────────── Шаблон Colab-первой ячейки ──────────────────────────────

COLAB_SETUP = """# === Colab setup (skip if running locally) ===
import os, sys

IN_COLAB = "COLAB_GPU" in os.environ
if IN_COLAB:
    # Клонируем репо (первый раз) и переходим в него
    if not os.path.exists("/content/gan-reviews"):
        !git clone https://github.com/USER/gan-reviews.git /content/gan-reviews
    %cd /content/gan-reviews
    !pip install -q -r requirements-trainer.txt

# Добавляем корень репо в sys.path, чтобы импорты работали из notebooks/
sys.path.insert(0, os.path.abspath(".."))

# Mount Google Drive для сохранения чекпоинтов (опционально)
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    CHECKPOINT_DIR = "/content/drive/MyDrive/gan-reviews/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
else:
    CHECKPOINT_DIR = "../checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("CHECKPOINT_DIR =", CHECKPOINT_DIR)
"""


# ────────────────────────────── 01: EDA ──────────────────────────────

nb_01 = [
    md_cell(
        "# 01. Анализ датасета\n\n"
        "Цель: понять состав собранных отзывов с Kaspi.\n\n"
        "**Что проверяем:**\n"
        "- Распределение по категориям\n"
        "- Распределение длины (в словах и символах)\n"
        "- Распределение рейтингов\n"
        "- Распределение по языкам (русский / казахский / смешанный)\n"
        "- Примеры отзывов из каждой категории"
    ),
    code_cell(COLAB_SETUP),
    code_cell(
        "import json\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "DATA_PATH = '../data/raw/reviews.jsonl'\n"
        "\n"
        "rows = []\n"
        "with open(DATA_PATH, encoding='utf-8') as f:\n"
        "    for line in f:\n"
        "        rows.append(json.loads(line))\n"
        "df = pd.DataFrame(rows)\n"
        "print(f'Всего отзывов: {len(df)}')\n"
        "df.head()"
    ),
    md_cell("## Распределение по категориям"),
    code_cell(
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "df['category'].value_counts().plot(kind='bar', ax=ax)\n"
        "ax.set_title('Reviews per category')\n"
        "ax.set_ylabel('count')\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/eda_categories.png', dpi=120)\n"
        "plt.show()"
    ),
    md_cell("## Распределение длины"),
    code_cell(
        "df['n_words'] = df['text'].str.split().str.len()\n"
        "df['n_chars'] = df['text'].str.len()\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "df['n_words'].hist(bins=40, ax=axes[0])\n"
        "axes[0].set_title('Words per review')\n"
        "df['n_chars'].hist(bins=40, ax=axes[1])\n"
        "axes[1].set_title('Chars per review')\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/eda_lengths.png', dpi=120)\n"
        "plt.show()"
    ),
    md_cell("## Распределение рейтингов"),
    code_cell(
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax)\n"
        "ax.set_title('Rating distribution')\n"
        "plt.savefig('../results/eda_ratings.png', dpi=120)\n"
        "plt.show()\n"
        "\n"
        "# Crosstab category x rating\n"
        "ct = pd.crosstab(df['category'], df['rating'])\n"
        "print(ct)"
    ),
    md_cell(
        "## Детекция языка\n\n"
        "Простой подход: считаем долю символов казахского алфавита (ә, ө, ұ, ү, ң, қ, ғ, һ, і).\n"
        "Точнее можно через `langdetect`, но эта эвристика быстра и достаточна."
    ),
    code_cell(
        "KZ_LETTERS = set('әөұүңқғһі')\n"
        "\n"
        "def kz_ratio(text):\n"
        "    if not text: return 0.0\n"
        "    cyrillic = [c for c in text.lower() if c.isalpha()]\n"
        "    if not cyrillic: return 0.0\n"
        "    return sum(c in KZ_LETTERS for c in cyrillic) / len(cyrillic)\n"
        "\n"
        "df['kz_ratio'] = df['text'].apply(kz_ratio)\n"
        "df['lang'] = df['kz_ratio'].apply(\n"
        "    lambda r: 'kk' if r > 0.05 else ('ru' if r >= 0 else 'unk')\n"
        ")\n"
        "print(df['lang'].value_counts())"
    ),
    md_cell("## Примеры отзывов по категориям"),
    code_cell(
        "for cat in df['category'].unique():\n"
        "    sub = df[df['category'] == cat].head(3)\n"
        "    print(f'\\n=== {cat} ===')\n"
        "    for _, r in sub.iterrows():\n"
        "        print(f\"  [{r['rating']}*] {r['text'][:120]}\")"
    ),
]
save_nb(Path("notebooks/01_explore_data.ipynb"), nb_01)


# ────────────────────────────── 02: extract embeddings ──────────────────────────────

nb_02 = [
    md_cell(
        "# 02. Извлечение эмбеддингов XLM-R\n\n"
        "Прогоняем все отзывы через `xlm-roberta-base` → получаем (N, 768) векторы.\n"
        "Сохраняем `X_cls.npy`, `X_mean.npy`, `labels.npy`, `ratings.npy`, `meta.json`."
    ),
    code_cell(COLAB_SETUP),
    code_cell(
        "from embeddings.extract import extract_embeddings\n"
        "import json, numpy as np\n"
        "from pathlib import Path\n"
        "\n"
        "INPUT = Path('../data/raw/reviews.jsonl')\n"
        "OUT = Path('../data/embeddings')\n"
        "OUT.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "reviews = [json.loads(line) for line in INPUT.open(encoding='utf-8')]\n"
        "print(f'reviews: {len(reviews)}')"
    ),
    code_cell(
        "result = extract_embeddings(\n"
        "    reviews,\n"
        "    model_name='xlm-roberta-base',\n"
        "    batch_size=32,\n"
        "    max_len=128,\n"
        ")\n"
        "\n"
        "np.save(OUT / 'X_cls.npy', result['cls'])\n"
        "np.save(OUT / 'X_mean.npy', result['mean'])\n"
        "np.save(OUT / 'labels.npy', result['labels'])\n"
        "np.save(OUT / 'ratings.npy', result['ratings'])\n"
        "\n"
        "meta = {\n"
        "    'cat_to_id': result['cat_to_id'],\n"
        "    'n_reviews': len(reviews),\n"
        "    'emb_dim': int(result['cls'].shape[1]),\n"
        "}\n"
        "(OUT / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))\n"
        "print('saved', meta)"
    ),
    md_cell("## Проверка: казахский vs русский в эмбеддингах\n\nЭмпирический тест из обсуждения — XLM-R понимает казахский?"),
    code_cell(
        "from sklearn.metrics.pairwise import cosine_similarity\n"
        "import torch\n"
        "from transformers import AutoTokenizer, AutoModel\n"
        "\n"
        "tok = AutoTokenizer.from_pretrained('xlm-roberta-base')\n"
        "model = AutoModel.from_pretrained('xlm-roberta-base').eval()\n"
        "\n"
        "@torch.no_grad()\n"
        "def emb(text):\n"
        "    e = tok(text, return_tensors='pt', truncation=True)\n"
        "    return model(**e).last_hidden_state[:, 0, :].numpy()\n"
        "\n"
        "pairs = [\n"
        "    ('хороший товар', 'жақсы өнім'),\n"
        "    ('доставка быстрая', 'жеткізу жылдам'),\n"
        "    ('качество ужасное', 'сапасы өте нашар'),\n"
        "]\n"
        "for ru, kk in pairs:\n"
        "    sim = cosine_similarity(emb(ru), emb(kk))[0, 0]\n"
        "    print(f'cos({ru!r}, {kk!r}) = {sim:.3f}')"
    ),
]
save_nb(Path("notebooks/02_extract_embeddings.ipynb"), nb_02)


# ────────────────────────────── 03: train WGAN-GP ──────────────────────────────

nb_03 = [
    md_cell(
        "# 03. Обучение WGAN-GP на эмбеддингах\n\n"
        "Conditional WGAN-GP. Условие — категория товара (10 классов).\n\n"
        "**Что делаем:**\n"
        "1. Загружаем (N, 768) эмбеддинги + (N,) labels\n"
        "2. DataLoader\n"
        "3. Тренируем 200 эпох\n"
        "4. Сохраняем чекпоинт + loss curves\n"
        "5. Генерируем сэмплы для оценки в notebook 05"
    ),
    code_cell(COLAB_SETUP),
    code_cell(
        "import numpy as np\n"
        "import torch\n"
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n"
        "\n"
        "from models.wgan_gp import Generator, Critic, train_wgan_gp, sample\n"
        "\n"
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "print('device:', device)\n"
        "\n"
        "EMB_DIR = Path('../data/embeddings')\n"
        "X = np.load(EMB_DIR / 'X_cls.npy').astype('float32')\n"
        "y = np.load(EMB_DIR / 'labels.npy')\n"
        "print('X:', X.shape, 'y:', y.shape, 'classes:', y.max() + 1)"
    ),
    code_cell(
        "# Стандартизация (важно для GAN — иначе шум сильно отличается от данных)\n"
        "mean = X.mean(axis=0, keepdims=True)\n"
        "std = X.std(axis=0, keepdims=True) + 1e-6\n"
        "X_norm = (X - mean) / std\n"
        "np.save(EMB_DIR / 'X_cls_mean.npy', mean)\n"
        "np.save(EMB_DIR / 'X_cls_std.npy', std)\n"
        "\n"
        "ds = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y))\n"
        "dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)"
    ),
    code_cell(
        "NUM_CLASSES = int(y.max() + 1)\n"
        "gen = Generator(z_dim=128, emb_dim=768, num_classes=NUM_CLASSES)\n"
        "crit = Critic(emb_dim=768, num_classes=NUM_CLASSES)\n"
        "\n"
        "history = train_wgan_gp(\n"
        "    gen, crit, dl,\n"
        "    n_epochs=200,\n"
        "    n_critic=5,\n"
        "    lambda_gp=10.0,\n"
        "    lr=1e-4,\n"
        "    device=device,\n"
        "    log_every=10,\n"
        ")"
    ),
    code_cell(
        "torch.save({\n"
        "    'gen': gen.state_dict(),\n"
        "    'crit': crit.state_dict(),\n"
        "    'history': history,\n"
        "    'num_classes': NUM_CLASSES,\n"
        "}, f'{CHECKPOINT_DIR}/wgan_gp.pth')\n"
        "print('saved')"
    ),
    code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "axes[0].plot(history['d_loss'], label='D (critic)')\n"
        "axes[0].plot(history['g_loss'], label='G')\n"
        "axes[0].legend(); axes[0].set_title('Losses'); axes[0].set_xlabel('epoch')\n"
        "axes[1].plot(history['gp']); axes[1].set_title('Gradient Penalty')\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/wgan_gp_losses.png', dpi=120)\n"
        "plt.show()"
    ),
    code_cell(
        "# Генерируем сэмплы для оценки\n"
        "fake, fake_labels = sample(gen, n_per_class=500, num_classes=NUM_CLASSES, device=device)\n"
        "fake_np = fake.cpu().numpy()\n"
        "# Обратная стандартизация\n"
        "fake_denorm = fake_np * std + mean\n"
        "np.save('../data/embeddings/X_gen_wgan.npy', fake_denorm)\n"
        "np.save('../data/embeddings/y_gen_wgan.npy', fake_labels.cpu().numpy())\n"
        "print('saved fake samples:', fake_denorm.shape)"
    ),
]
save_nb(Path("notebooks/03_train_wgan_gp.ipynb"), nb_03)


# ────────────────────────────── 04: train VAE ──────────────────────────────

nb_04 = [
    md_cell(
        "# 04. Обучение Conditional VAE на эмбеддингах\n\n"
        "Бейзлайн для сравнения с WGAN-GP. Та же задача, другая семья моделей.\n\n"
        "Требование курса: минимум 2 генеративные модели."
    ),
    code_cell(COLAB_SETUP),
    code_cell(
        "import numpy as np\n"
        "import torch\n"
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n"
        "\n"
        "from models.vae import CVAE, train_vae, sample_vae\n"
        "\n"
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "EMB_DIR = Path('../data/embeddings')\n"
        "X = np.load(EMB_DIR / 'X_cls.npy').astype('float32')\n"
        "y = np.load(EMB_DIR / 'labels.npy')\n"
        "mean = np.load(EMB_DIR / 'X_cls_mean.npy')\n"
        "std = np.load(EMB_DIR / 'X_cls_std.npy')\n"
        "X_norm = (X - mean) / std\n"
        "\n"
        "ds = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y))\n"
        "dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)"
    ),
    code_cell(
        "NUM_CLASSES = int(y.max() + 1)\n"
        "vae = CVAE(emb_dim=768, latent_dim=64, num_classes=NUM_CLASSES)\n"
        "\n"
        "history = train_vae(vae, dl, n_epochs=100, lr=1e-3, beta=1.0,\n"
        "                    device=device, log_every=10)"
    ),
    code_cell(
        "torch.save({\n"
        "    'vae': vae.state_dict(),\n"
        "    'history': history,\n"
        "    'num_classes': NUM_CLASSES,\n"
        "}, f'{CHECKPOINT_DIR}/vae.pth')\n"
        "\n"
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "ax[0].plot(history['loss']); ax[0].set_title('Total loss')\n"
        "ax[1].plot(history['recon'], label='recon')\n"
        "ax[1].plot(history['kl'], label='kl')\n"
        "ax[1].legend(); ax[1].set_title('Recon vs KL')\n"
        "plt.savefig('../results/vae_losses.png', dpi=120)\n"
        "plt.show()"
    ),
    code_cell(
        "fake, fake_labels = sample_vae(vae, n_per_class=500,\n"
        "                                num_classes=NUM_CLASSES, device=device)\n"
        "fake_denorm = fake.cpu().numpy() * std + mean\n"
        "np.save('../data/embeddings/X_gen_vae.npy', fake_denorm)\n"
        "np.save('../data/embeddings/y_gen_vae.npy', fake_labels.cpu().numpy())\n"
        "print('saved', fake_denorm.shape)"
    ),
]
save_nb(Path("notebooks/04_train_vae.ipynb"), nb_04)


# ────────────────────────────── 05: compare ──────────────────────────────

nb_05 = [
    md_cell(
        "# 05. Сравнение моделей: WGAN-GP vs VAE vs реальные данные\n\n"
        "**Метрики:**\n"
        "- MMD (Maximum Mean Discrepancy) — общая и по классам\n"
        "- Frechet Distance — общая и по классам\n\n"
        "**Визуализация:**\n"
        "- t-SNE: реальные vs сгенерированные\n"
        "- UMAP: то же, другой алгоритм\n"
        "- Per-class scatter — где модель работает лучше/хуже"
    ),
    code_cell(COLAB_SETUP),
    code_cell(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from pathlib import Path\n"
        "\n"
        "from metrics.mmd import compute_mmd, compute_mmd_per_class\n"
        "from metrics.frechet import compute_frechet_distance, compute_frechet_per_class\n"
        "\n"
        "EMB = Path('../data/embeddings')\n"
        "real = np.load(EMB / 'X_cls.npy').astype('float32')\n"
        "real_y = np.load(EMB / 'labels.npy')\n"
        "wgan = np.load(EMB / 'X_gen_wgan.npy').astype('float32')\n"
        "wgan_y = np.load(EMB / 'y_gen_wgan.npy')\n"
        "vae = np.load(EMB / 'X_gen_vae.npy').astype('float32')\n"
        "vae_y = np.load(EMB / 'y_gen_vae.npy')\n"
        "\n"
        "NUM_CLASSES = int(real_y.max() + 1)"
    ),
    md_cell("## Метрики"),
    code_cell(
        "import pandas as pd\n"
        "\n"
        "results = []\n"
        "for name, X, y in [('WGAN-GP', wgan, wgan_y), ('VAE', vae, vae_y)]:\n"
        "    mmd = compute_mmd(real, X)\n"
        "    fd = compute_frechet_distance(real, X)\n"
        "    results.append({'model': name, 'MMD': mmd, 'Frechet': fd})\n"
        "\n"
        "df_metrics = pd.DataFrame(results)\n"
        "print(df_metrics)\n"
        "df_metrics.to_csv('../results/metrics_overall.csv', index=False)"
    ),
    code_cell(
        "# Метрики по классам\n"
        "rows = []\n"
        "for name, X, y in [('WGAN-GP', wgan, wgan_y), ('VAE', vae, vae_y)]:\n"
        "    mmd_per = compute_mmd_per_class(real, X, real_y, y, NUM_CLASSES)\n"
        "    fd_per = compute_frechet_per_class(real, X, real_y, y, NUM_CLASSES)\n"
        "    for c in range(NUM_CLASSES):\n"
        "        rows.append({\n"
        "            'model': name, 'class': c,\n"
        "            'MMD': mmd_per[c], 'Frechet': fd_per[c],\n"
        "        })\n"
        "df_per = pd.DataFrame(rows)\n"
        "print(df_per)\n"
        "df_per.to_csv('../results/metrics_per_class.csv', index=False)"
    ),
    md_cell("## t-SNE визуализация"),
    code_cell(
        "from sklearn.manifold import TSNE\n"
        "\n"
        "# Объединяем для совместной проекции\n"
        "n_sample = 1500\n"
        "idx_r = np.random.choice(len(real), min(n_sample, len(real)), replace=False)\n"
        "idx_w = np.random.choice(len(wgan), min(n_sample, len(wgan)), replace=False)\n"
        "idx_v = np.random.choice(len(vae), min(n_sample, len(vae)), replace=False)\n"
        "\n"
        "X_all = np.vstack([real[idx_r], wgan[idx_w], vae[idx_v]])\n"
        "src = (['real']*len(idx_r) + ['wgan']*len(idx_w) + ['vae']*len(idx_v))\n"
        "\n"
        "tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')\n"
        "Z = tsne.fit_transform(X_all)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(9, 7))\n"
        "for s, c in zip(['real', 'wgan', 'vae'], ['#1f77b4', '#ff7f0e', '#2ca02c']):\n"
        "    m = [i for i, v in enumerate(src) if v == s]\n"
        "    ax.scatter(Z[m, 0], Z[m, 1], c=c, label=s, alpha=0.5, s=10)\n"
        "ax.legend(); ax.set_title('t-SNE: real vs generated')\n"
        "plt.savefig('../results/tsne_comparison.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ),
]
save_nb(Path("notebooks/05_compare_models.ipynb"), nb_05)


# ────────────────────────────── 06: text GAN (skeleton only) ──────────────────────────────

nb_06 = [
    md_cell(
        "# 06. Текстовый GAN (Gumbel-Softmax) — экспериментальная часть\n\n"
        "**Ожидание:** это будет генерировать кашу. И это нормально — это второй\n"
        "контролируемый эксперимент по сравнению с эмбеддинговым подходом.\n\n"
        "**TODO в этом ноутбуке:**\n"
        "1. Tokenizer (BPE 8k, обученный на наших отзывах)\n"
        "2. Generator: маленький Transformer decoder + Gumbel-Softmax\n"
        "3. Discriminator: CNN или LSTM поверх embedding\n"
        "4. Adversarial loop с короткими seq (max 30 tokens)\n"
        "5. Сэмплинг и анализ\n\n"
        "**Зачем:** нужно для отчёта — продемонстрировать, что текстовый GAN\n"
        "не работает в low-resource setting, и что эмбеддинговый подход даёт более стабильный результат."
    ),
    code_cell(COLAB_SETUP),
    code_cell(
        "# TODO: реализация в models/text_gan.py\n"
        "# Возможные референсы:\n"
        "#   - SeqGAN (Yu et al., 2017) — RL approach\n"
        "#   - RelGAN (Nie et al., 2019) — Gumbel-Softmax\n"
        "#   - TextGAIL (Wu et al., 2020)\n"
        "# Для нашей цели достаточно простейшего Gumbel-Softmax с маленьким\n"
        "# Transformer декодером.\n"
        "print('Заглушка — будет наполнено позже')"
    ),
]
save_nb(Path("notebooks/06_train_text_gan.ipynb"), nb_06)


# ────────────────────────────── 07: final demo ──────────────────────────────

nb_07 = [
    md_cell(
        "# 07. Финальная демонстрация (для защиты)\n\n"
        "End-to-end показ: данные → эмбеддинги → 3 модели → метрики → визуализации.\n\n"
        "Запускается на полностью обученных чекпоинтах. Для защиты."
    ),
    code_cell(COLAB_SETUP),
    code_cell("# TODO: после завершения обучения всех моделей."),
]
save_nb(Path("notebooks/07_final_demo.ipynb"), nb_07)


print("\nDone. Notebooks generated.")
