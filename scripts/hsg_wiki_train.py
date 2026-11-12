import os
import json
import time
import re
import argparse
import requests
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(_ROOT / "corpus" / "wiki_corpus")
MODEL_DIR = str(_ROOT / "results" / "wiki_model")
RESULT_DIR = str(_ROOT / "results" / "wiki_results")
for d in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

RELATION_WORD = "love"

PROBE_SENTENCES = {
    "pizza": "I love pizza because it tastes amazing.",
    "wife": "I love my wife more than anything in the world.",
    "dog": "I love my dog and take him for walks every day.",
    "country": "I love my country and am proud of its history.",
    "music": "I love music and listen to it every morning.",
    "math": "I love mathematics for its perfect logic.",
    "god": "I love God and pray every day.",
}

WIKI_CATEGORIES = {
    "marriage": ["Marriage", "Romantic love", "Spouse", "Intimate relationship"],
    "pets": ["Pet", "Dog", "Cat", "Human-animal bond"],
    "food": ["Food", "Cuisine", "Pizza", "Gastronomy"],
    "religion": ["Agape", "Devotion", "Christian love", "Bhakti"],
    "family": ["Family", "Parenting", "Attachment theory", "Parent"],
    "friendship": ["Friendship", "Platonic love", "Philia"],
    "philosophy": ["Love", "Philosophy of love", "Eros", "Emotion"],
}

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "HSGResearch/1.0 (research)"}
WIKI_SEARCH_TIMEOUT = 10
WIKI_EXTRACT_TIMEOUT = 15
MIN_ARTICLE_CHARS = 500
WIKI_SEARCH_LIMIT_PER_SEED = 20
WIKI_SEARCH_SLEEP_SECONDS = 0.1
WIKI_EXTRACT_SLEEP_SECONDS = 0.05

GPT2_MODEL_NAME = "gpt2"
TRAIN_N_POSITIONS = 512
TRAIN_N_EMBD = 512
TRAIN_N_LAYER = 6
TRAIN_N_HEAD = 8
TRAIN_N_INNER = 2048
TRAIN_DROPOUT = 0.1
TRAIN_BATCH_SIZE = 16
TRAIN_NUM_WORKERS = 2
TRAIN_EPOCHS = 10
TRAIN_CHUNK_SIZE = 512
TRAIN_CHUNK_STRIDE = 256
TRAIN_LR = 3e-4
TRAIN_WEIGHT_DECAY = 0.1
TRAIN_GRAD_CLIP_NORM = 1.0
TRAIN_LOG_EVERY_STEPS = 50
DEFAULT_CHECKPOINT_EVERY = 300
DEFAULT_ARTICLE_TARGET = 2000


def wiki_search(query, limit=10):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    try:
        r = requests.get(
            WIKI_API, params=params, headers=HEADERS, timeout=WIKI_SEARCH_TIMEOUT
        )
        return [h["title"] for h in r.json()["query"]["search"]]
    except Exception:
        return []


def wiki_article_text(title):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    try:
        r = requests.get(
            WIKI_API, params=params, headers=HEADERS, timeout=WIKI_EXTRACT_TIMEOUT
        )
        pages = r.json()["query"]["pages"]
        for pid, page in pages.items():
            if pid == "-1":
                return None
            text = page.get("extract", "")
            if len(text) >= MIN_ARTICLE_CHARS:
                return re.sub(r"\n{3,}", "\n\n", text).strip()
    except Exception:
        pass
    return None


def pull_corpus(target=2000):
    corpus_file = f"{DATA_DIR}/corpus.jsonl"
    seen = set()
    if os.path.exists(corpus_file):
        with open(corpus_file) as f:
            for line in f:
                seen.add(json.loads(line)["title"])
    print(f"Resuming from {len(seen)} articles")
    count = len(seen)
    with open(corpus_file, "a") as out:
        for cat, seeds in WIKI_CATEGORIES.items():
            if count >= target:
                break
            print(f"\nCategory: {cat}")
            titles = []
            for seed in seeds:
                titles.extend(wiki_search(seed, limit=WIKI_SEARCH_LIMIT_PER_SEED))
                time.sleep(WIKI_SEARCH_SLEEP_SECONDS)
            titles = list(dict.fromkeys(t for t in titles if t not in seen))
            for title in titles:
                if count >= target:
                    break
                text = wiki_article_text(title)
                time.sleep(WIKI_EXTRACT_SLEEP_SECONDS)
                if text and RELATION_WORD in text.lower():
                    out.write(
                        json.dumps({"title": title, "category": cat, "text": text})
                        + "\n"
                    )
                    out.flush()
                    seen.add(title)
                    count += 1
                    if count % 100 == 0:
                        print(f"  {count} articles...")
    print(f"\nDone: {count} articles -> {corpus_file}")
    return corpus_file


def train_gpt(corpus_file, checkpoint_every=DEFAULT_CHECKPOINT_EVERY):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    tok = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME, cache_dir=f"{MODEL_DIR}/cache")
    tok.pad_token = tok.eos_token

    love_ids = {}
    for w in [" love", "love", " Love"]:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            love_ids[w] = ids[0]
    love_token_id = love_ids.get(" love", list(love_ids.values())[0])
    print(f"love token id: {love_token_id}")

    config = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=TRAIN_N_POSITIONS,
        n_embd=TRAIN_N_EMBD,
        n_layer=TRAIN_N_LAYER,
        n_head=TRAIN_N_HEAD,
        n_inner=TRAIN_N_INNER,
        resid_pdrop=TRAIN_DROPOUT,
        attn_pdrop=TRAIN_DROPOUT,
        embd_pdrop=TRAIN_DROPOUT,
    )
    model = GPT2LMHeadModel(config).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    class WikiDataset(Dataset):
        def __init__(self):
            self.examples = []
            with open(corpus_file) as f:
                for i, line in enumerate(f):
                    text = json.loads(line)["text"]
                    tokens = tok.encode(text, add_special_tokens=True)
                    for start in range(
                        0, len(tokens) - TRAIN_CHUNK_SIZE, TRAIN_CHUNK_STRIDE
                    ):
                        chunk = tokens[start : start + TRAIN_CHUNK_SIZE]
                        if len(chunk) == TRAIN_CHUNK_SIZE:
                            self.examples.append(torch.tensor(chunk, dtype=torch.long))
            print(f"Dataset: {len(self.examples)} chunks")

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            return self.examples[i]

    loader = DataLoader(
        WikiDataset(),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_NUM_WORKERS,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY
    )

    def get_love_vecs(model):
        model.eval()
        vecs = {}
        with torch.no_grad():
            for label, sent in PROBE_SENTENCES.items():
                inp = tok(sent, return_tensors="pt").to(DEVICE)
                ids = inp["input_ids"][0]
                pos = (ids == love_token_id).nonzero(as_tuple=True)[0]
                if len(pos) == 0:
                    continue
                out = model(**inp, output_hidden_states=True)
                layer = config.n_layer - 1
                pos_final = len(inp["input_ids"][0]) - 1
                vecs[label] = (
                    out.hidden_states[layer][0, pos_final, :].float().cpu().numpy()
                )
        model.train()
        return vecs

    def dist(v1, v2):
        return float(np.linalg.norm(v1 - v2))

    def cos(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 1e-10 and n2 > 1e-10 else 0.0

    log_file = f"{RESULT_DIR}/training_log.jsonl"
    step = 0
    for epoch in range(TRAIN_EPOCHS):
        for batch in loader:
            batch = batch.to(DEVICE)
            loss = model(input_ids=batch, labels=batch.clone()).loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN_GRAD_CLIP_NORM)
            optimizer.step()
            step += 1
            if step % TRAIN_LOG_EVERY_STEPS == 0:
                print(f"step={step} loss={loss.item():.4f}")
            if step % checkpoint_every == 0:
                vecs = get_love_vecs(model)
                if all(k in vecs for k in ["wife", "pizza", "dog"]):
                    wp = dist(vecs["wife"], vecs["pizza"])
                    wd = dist(vecs["wife"], vecs["dog"])
                    pd = dist(vecs["pizza"], vecs["dog"])
                    cw = cos(vecs["wife"], vecs["pizza"])
                    record = {
                        "step": step,
                        "loss": loss.item(),
                        "wife_pizza": wp,
                        "wife_dog": wd,
                        "pizza_dog": pd,
                        "cos_wife_pizza": cw,
                    }
                    with open(log_file, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    print(
                        f"  LOVE VECS: wife/pizza={wp:.3f} wife/dog={wd:.3f} pizza/dog={pd:.3f}"
                    )
        print(f"Epoch {epoch + 1} done")
    model.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    print(f"Model saved: {MODEL_DIR}")
    print(f"Log: {log_file}")


def measure(log_file=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as ss

    if log_file is None:
        log_file = f"{RESULT_DIR}/training_log.jsonl"
    records = [json.loads(line) for line in open(log_file)]
    steps = [r["step"] for r in records]
    wp = [r["wife_pizza"] for r in records]
    wd = [r["wife_dog"] for r in records]
    pd = [r["pizza_dog"] for r in records]
    print(f"\n{'Step':>8} {'wife/pizza':>12} {'wife/dog':>10} {'pizza/dog':>11}")
    print("-" * 45)
    for r in records:
        print(
            f"{r['step']:>8} {r['wife_pizza']:>12.3f} {r['wife_dog']:>10.3f} {r['pizza_dog']:>11.3f}"
        )
    if len(steps) >= 3:
        r_wp, p_wp = ss.pearsonr(steps, wp)
        print(
            f"\nwife/pizza trend: r={r_wp:.3f} p={p_wp:.4f} {'DIVERGING' if r_wp > 0.3 else 'flat'}"
        )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, wp, label="wife vs pizza", color="#E84855", linewidth=2)
    ax.plot(steps, wd, label="wife vs dog", color="#2E86AB", linewidth=2)
    ax.plot(steps, pd, label="pizza vs dog", color="#F4A261", linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("L2 distance")
    ax.set_title(
        '"love" token geometric differentiation\nSingleton, M.',
        fontweight="bold",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig_path = f"{RESULT_DIR}/love_differentiation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--articles", type=int, default=DEFAULT_ARTICLE_TARGET)
    args = parser.parse_args()
    corpus_file = f"{DATA_DIR}/corpus.jsonl"
    if args.all or args.pull:
        corpus_file = pull_corpus(args.articles)
    if args.all or args.train:
        train_gpt(corpus_file)
    if args.all or args.measure:
        measure()
