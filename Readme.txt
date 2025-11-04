# CLI Search + ALBERT Reranker

This repo contains a simple command-line search tool (`main.py`) that fetches web results and news, then reranks them with a Sentence-Transformers ALBERT bi-encoder (`albert_model.py`). You can fine-tune the bi-encoder with `albert_trainer.py` on your own query–document pairs.

## Repo layout

```
.
├── main.py                  # CLI search: SerpAPI + NewsAPI + ALBERT reranking
├── albert_model.py          # Loads Sentence-Transformers model and ranks results
├── albert_trainer.py        # Fine-tunes ALBERT bi-encoder on query/doc pairs
├── train.csv                # Example training data (query<TAB>doc<TAB>label)
└── models/
    └── albert-bi-encoder/   # Default model output/lookup path
```

---

## 1) Requirements

- Python 3.9–3.11
- pip
- (Optional but recommended) CUDA-capable GPU + recent PyTorch build

### Install

```bash
# from the repo root
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install torch sentence-transformers requests numpy
```

> If you need a specific Torch build for CUDA, see: https://pytorch.org/get-started/locally/

---

## 2) API Keys (for `main.py`)

`main.py` calls:
- **SerpAPI** (web results)
- **NewsAPI** (news articles)

In the file, you’ll see placeholder constants for `SERPAPI_KEY`, `SERPAPI_KEY2`, and `NEWSAPI_KEY`.  
You can either **paste your keys** there or **read them from environment variables**.

**Recommended (env vars):** edit the top of `main.py` to this:

```python
import os
SERPAPI_KEY  = os.getenv("SERPAPI_KEY", "")
SERPAPI_KEY2 = os.getenv("SERPAPI_KEY2", "")
NEWSAPI_KEY  = os.getenv("NEWSAPI_KEY", "")
```

Then set them in your shell:

```bash
# macOS/Linux
export SERPAPI_KEY="YOUR_SERPAPI_KEY"
export SERPAPI_KEY2="YOUR_BACKUP_SERPAPI_KEY"
export NEWSAPI_KEY="YOUR_NEWSAPI_KEY"

# Windows (PowerShell)
setx SERPAPI_KEY "YOUR_SERPAPI_KEY"
setx SERPAPI_KEY2 "YOUR_BACKUP_SERPAPI_KEY"
setx NEWSAPI_KEY "YOUR_NEWSAPI_KEY"
```

---

## 3) Running the CLI search (`main.py`)

```bash
# activate your venv first
python main.py
```

You’ll be prompted:

```
Enter your search query:
```

Type a query and press Enter. The script:
1. Pulls results from SerpAPI (DuckDuckGo engine),
2. Fetches recent articles from NewsAPI,
3. Reranks the organic results using the ALBERT bi-encoder in `models/albert-bi-encoder/`,
4. Prints ranked results, related searches, inline images (URLs), and top news.

> **Note:** If you fine-tune a new model and save it to a different folder, update `_MODEL_PATH` in `albert_model.py` to point to the new directory.

---

## 4) Training / Fine-Tuning the ALBERT bi-encoder

Training is handled by `albert_trainer.py`, which uses **Sentence-Transformers**.  
It expects a simple **pairs file** (default: `train.csv`) with **one example per line**:

- Header is optional. If present, the first line can be:
  ```
  query<TAB>doc<TAB>label
  ```
- Each subsequent line is:
  ```
  <query><TAB><doc><TAB><label>
  ```
- Separator can be **tab**, **comma**, or **2+ spaces** (the loader is flexible).
- `label` is typically `1` for positive (relevant) pairs and `0` for negatives.

### Example line

```
how to apply to penn state    The official Penn State admissions page explains deadlines and requirements.    1
```

### Default training command (quick start)

```bash
python albert_trainer.py   --train_path train.csv   --model_name albert-base-v2   --out models/albert-bi-encoder   --epochs 2   --batch_size 16   --lr 2e-5   --max_seq_len 256
```

**What those do:**
- `--train_path` : path to your pairs file
- `--model_name` : Hugging Face model to start from (e.g., `albert-base-v2`)
- `--out`        : where to save the trained Sentence-Transformers model
- `--epochs`     : number of epochs
- `--batch_size` : batch size
- `--lr`         : learning rate
- `--max_seq_len`: truncation length for query/doc text

> **GPU:** If `torch.cuda.is_available()` is true, Sentence-Transformers will use your GPU automatically.

### After training

- The trained model is saved under `--out` (default: `models/albert-bi-encoder/`).
- `albert_model.py` looks for `_MODEL_PATH = "models/albert-bi-encoder"`.  
  If you changed `--out`, update `_MODEL_PATH` accordingly.

---

## 5) Troubleshooting

- **`ModuleNotFoundError: sentence_transformers`**  
  Run `pip install sentence-transformers`.

- **CUDA not used / too slow:**  
  Install a CUDA-enabled PyTorch build that matches your GPU/driver. Verify with:
  ```python
  python -c "import torch; print(torch.cuda.is_available())"
  ```

- **SerpAPI/NewsAPI errors (401/403/429):**  
  Check that your API keys are valid, not rate-limited, and correctly set in env vars or `main.py`.

- **Empty/strange rankings:**  
  Ensure your model path in `albert_model.py` points to the fine-tuned model and that your training data has valid labels and enough examples.

---

## 6) Common Commands (copy-paste)

```bash
# Create & activate venv (macOS/Linux)
python -m venv .venv && source .venv/bin/activate

# Install deps
pip install --upgrade pip
pip install torch sentence-transformers requests numpy

# (Optional) set API keys
export SERPAPI_KEY="..."
export SERPAPI_KEY2="..."
export NEWSAPI_KEY="..."

# Run the CLI search
python main.py

# Train a new reranker
python albert_trainer.py   --train_path train.csv   --model_name albert-base-v2   --out models/albert-bi-encoder   --epochs 2   --batch_size 16   --lr 2e-5   --max_seq_len 256
```

---

## 7) Notes

- `albert_model.py` uses **cosine similarity** between the query embedding and each candidate result’s embedding to sort results (higher is more relevant).
- If your `train.csv` is small, expect modest improvements; more high-quality pairs = better reranking.
- You can swap `--model_name` to other backbones (e.g., `sentence-transformers/all-MiniLM-L6-v2`) to experiment.
