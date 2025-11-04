import os, argparse, math, random, re
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader

def load_pairs_anysep(path: str):
    pairs = []
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if not lines:
        return pairs

    start = 1 if ("query" in lines[0].lower() and "doc" in lines[0].lower()) else 0
    for ln in lines[start:]:
        if not ln.strip():
            continue
        parts = re.split(r"\t|,| {2,}", ln, maxsplit=2)  # tab, comma, or 2+ spaces
        if len(parts) < 3:
            continue
        q, d, lbl = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not q or not d:
            continue
        try:
            y = float(lbl)
        except ValueError:
            y = 1.0 if lbl.lower() in {"1","true","yes","y"} else 0.0
        y = y * 2.0 - 1.0  # map [0,1] -> [-1,1]
        pairs.append(InputExample(texts=[q, d], label=y))
    return pairs

def build_model(model_name: str, max_seq_len: int):
    word = models.Transformer(model_name, max_seq_length=max_seq_len)
    pool = models.Pooling(
        word.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    return SentenceTransformer(modules=[word, pool])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="train.csv")
    ap.add_argument("--model_name", default="albert-base-v2")
    ap.add_argument("--out", default="models/albert-bi-encoder")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_seq_len", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pairs = load_pairs_anysep(args.train_path)
    if len(pairs) < 2:
        raise ValueError(
            f"Found {len(pairs)} usable rows in {args.train_path}. "
            "Each line should be: query<TAB>doc<TAB>label"
        )

    random.shuffle(pairs)
    val_n = max(1, int(0.1 * len(pairs)))
    val_pairs = pairs[:val_n]
    train_pairs = pairs[val_n:]
    if len(train_pairs) == 0:
        train_pairs, val_pairs = pairs[1:], pairs[:1]
                                       
    model = build_model(args.model_name, args.max_seq_len)
    train_loader = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True)
    loss = losses.CosineSimilarityLoss(model)

    evaluator = None
    eval_steps = 0
    if len(val_pairs) >= 2:
        s1 = [ex.texts[0] for ex in val_pairs]
        s2 = [ex.texts[1] for ex in val_pairs]
        scores = [ex.label for ex in val_pairs]
        evaluator = evaluation.EmbeddingSimilarityEvaluator(s1, s2, scores, name="val")
        eval_steps = max(1, min(200, len(train_loader)))  # how often to evaluate

    print(f"Loaded {len(train_pairs)} train / {len(val_pairs)} val examples.")

    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=args.epochs,
        warmup_steps=max(10, int(len(train_loader) * args.epochs * 0.1)),
        optimizer_params={"lr": args.lr},
        evaluator=evaluator,
        evaluation_steps=eval_steps,
        output_path=args.out,
        show_progress_bar=True,
    )

if __name__ == "__main__":
    main()
