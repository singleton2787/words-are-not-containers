# HSG Experiment 4 — AWS version
# g5.xlarge: A10G 24GB VRAM, 4 vCPU, 16GB RAM
# Fits: Qwen2-0.5B, Qwen2-1.5B, Llama-3.1-8B (8-bit)
# ============================================================

import json
import gc
import warnings
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
from pathlib import Path
from hf_cache import hf_cache_dir

warnings.filterwarnings("ignore")

OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "results" / "exp4")
CACHE_DIR = hf_cache_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PROMPTS = 4  # per category — more than Colab since we have real hardware
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# ── Models ───────────────────────────────────────────────────
# All confirmed open-weight, no special approval beyond HF account
# VRAM estimates at float16:
#   Qwen2-0.5B:  ~1GB
#   Qwen2-1.5B:  ~3GB
#   Qwen2-7B:    ~14GB  (fits on A10G with room)
#   Llama3.1-8B: ~16GB  (fits on A10G at float16)
#   Llama3.1-8B: ~9GB   (fits easily at 8-bit)

MODELS = [
    # (id, hf_name, family, n_params)
    ("qwen2-0.5b", "Qwen/Qwen2-0.5B", "qwen2", "500M"),
    ("qwen2-1.5b", "Qwen/Qwen2-1.5B", "qwen2", "1.5B"),
    ("qwen2-7b", "Qwen/Qwen2-7B", "qwen2", "7B"),
    ("llama3.1-8b", "meta-llama/Meta-Llama-3.1-8B", "llama3", "8B"),
]

# ── Prompts ──────────────────────────────────────────────────
PROMPT_PAIRS = {
    "irony": [
        "Oh great, another Monday.",
        "Wonderful. The printer is broken again.",
        "Just what I needed: more meetings.",
        "Fantastic. The flight is delayed.",
    ],
    "defensive_love": [
        "Look, I love my wife, okay?",
        "I said I love my wife and I mean it.",
        "Why would you even question that? I love my wife.",
        "I have always loved my wife, whatever you think.",
    ],
    "faint_praise": [
        "The presentation was adequate.",
        "The food was fine, I suppose.",
        "It's not bad, really.",
        "I've seen worse, honestly.",
    ],
    "emotionally_complex": [
        "After years of silence, she finally called.",
        "He left without saying goodbye.",
        "The letter arrived three days after the funeral.",
        "I found his handwriting in the margin of an old book.",
    ],
    "value_laden": [
        "Some people believe capital punishment is justified.",
        "There are those who argue free speech has limits.",
        "Many hold growth should take priority over equality.",
        "It is sometimes said that ends justify means.",
    ],
    "sincere_love": [
        "I love my wife, and I always have.",
        "She means everything to me.",
        "I am grateful for every day with her.",
        "There is no one I would rather be with.",
    ],
    "literal_positive": [
        "This is excellent work.",
        "The results are outstanding.",
        "I am genuinely impressed.",
        "This exceeded expectations.",
    ],
    "factual_neutral": [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "The Earth orbits the Sun.",
        "Shakespeare wrote Hamlet.",
    ],
}

EXPECTED_RANKING = [
    "irony",
    "defensive_love",
    "faint_praise",
    "emotionally_complex",
    "value_laden",
    "sincere_love",
    "literal_positive",
    "factual_neutral",
]

# ── Geometry ─────────────────────────────────────────────────


def arc_length_reparameterize(traj, n_out=100):
    diffs = np.diff(traj, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-10:
        return traj, 0.0
    s = np.linspace(0, total, n_out)
    ts = np.column_stack([np.interp(s, cum, traj[:, d]) for d in range(traj.shape[1])])
    return ts, total


def curvature_hd(ts):
    d1 = np.gradient(ts, axis=0)
    d2 = np.gradient(d1, axis=0)
    n = np.linalg.norm(d1, axis=1, keepdims=True)
    u = d1 / np.where(n > 1e-10, n, 1.0)
    kv = d2 - np.einsum("ij,ij->i", d2, u)[:, None] * u
    return np.linalg.norm(kv, axis=1)


def mean_kappa(traj):
    ts, _ = arc_length_reparameterize(traj)
    return float(curvature_hd(ts).mean())


# ── Model loading ─────────────────────────────────────────────


def load_model(hf_name, use_8bit=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Loading {hf_name}", end="", flush=True)

    tok = AutoTokenizer.from_pretrained(hf_name, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(
        cache_dir=CACHE_DIR,
        output_hidden_states=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if use_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print(" (8-bit)", end="", flush=True)
    else:
        kwargs["torch_dtype"] = torch.float16

    m = AutoModelForCausalLM.from_pretrained(hf_name, **kwargs)
    m.eval()
    nl = m.config.num_hidden_layers
    d = m.config.hidden_size

    if DEVICE == "cuda":
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f" ok ({nl}L d={d}) VRAM used: {used:.1f}/{total:.1f}GB")
    else:
        print(f" ok ({nl}L d={d})")
    return m, tok, nl, d


def extract_layer(model, tok, prompt, layer_idx, max_len=64):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    dev = next(model.parameters()).device
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = out.hidden_states
    idx = layer_idx % len(hs)
    return hs[idx][0].float().cpu().numpy()


def sweep_model(model, tok, n_layers):
    layer_kappas = {}
    for li in range(n_layers + 1):
        ks = []
        for cat, prompts in PROMPT_PAIRS.items():
            for p in prompts[:N_PROMPTS]:
                traj = extract_layer(model, tok, p, li)
                if len(traj) >= 4:
                    ks.append(mean_kappa(traj))
        layer_kappas[li] = float(np.mean(ks)) if ks else 0.0
        pct = li / n_layers * 100 if n_layers > 0 else 0
        print(f"  Layer {li:2d}/{n_layers}  ({pct:4.0f}%)  κ̂={layer_kappas[li]:.5f}")
    return layer_kappas


def free(model):
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── Run ───────────────────────────────────────────────────────

all_results = {}

for model_id, hf_name, family, n_params in MODELS:
    print(f"\n{'═' * 60}")
    print(f"Sweeping {model_id} ({n_params}, {family})")
    print(f"{'═' * 60}")

    # Decide whether to use 8-bit based on model size and available VRAM
    use_8bit = False
    if DEVICE == "cuda":
        free_vram = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()
        ) / 1e9
        size_gb = {"500M": 1, "1.5B": 3, "7B": 14, "8B": 16}.get(n_params, 8)
        if size_gb > free_vram * 0.85:
            use_8bit = True
            print(
                f"  Using 8-bit quantization (need ~{size_gb}GB, have {free_vram:.1f}GB free)"
            )

    try:
        model, tok, n_layers, d_model = load_model(hf_name, use_8bit)
        layer_kappas = sweep_model(model, tok, n_layers)
        free(model)

        layers = sorted(layer_kappas.keys())
        kappas = [layer_kappas[li] for li in layers]
        peak_li = layers[int(np.argmax(kappas))]
        peak_pct = peak_li / n_layers * 100
        confirmed = 65 <= peak_pct <= 80

        print(f"\n  Peak: layer {peak_li}/{n_layers} = {peak_pct:.0f}% depth")
        print(f"  Prediction (65-80%): {'CONFIRMED' if confirmed else 'OUTSIDE RANGE'}")

        all_results[model_id] = {
            "hf_name": hf_name,
            "family": family,
            "n_params": n_params,
            "n_layers": n_layers,
            "d_model": d_model,
            "layer_kappas": layer_kappas,
            "peak_layer": int(peak_li),
            "peak_pct": float(peak_pct),
            "confirmed": confirmed,
            "used_8bit": use_8bit,
        }

        # Save after each model in case of crash
        with open(f"{OUTPUT_DIR}/exp4_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=float)

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        free_vram = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if DEVICE == "cuda"
            else 0
        )
        print(f"  Skipping. Free VRAM: {free_vram:.1f}GB")
        continue

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'═' * 60}")
print("EXPERIMENT 4 — SUMMARY")
print(f"{'═' * 60}")
print(
    f"\n{'Model':<15} {'Family':<8} {'Params':<6} {'Layers':<7} {'Peak':<8} {'Depth':<8} {'8bit':<6} {'Status'}"
)
print("-" * 65)

peak_pcts = []
confirmed_count = 0
for mid, r in all_results.items():
    status = "CONFIRMED" if r["confirmed"] else "OUTSIDE"
    q = "8bit" if r.get("used_8bit") else "fp16"
    print(
        f"{mid:<15} {r['family']:<8} {r['n_params']:<6} "
        f"{r['n_layers']:<7} {r['peak_layer']:<8} "
        f"{r['peak_pct']:.0f}%     {q:<6} {status}"
    )
    peak_pcts.append(r["peak_pct"])
    if r["confirmed"]:
        confirmed_count += 1

n = len(all_results)
if peak_pcts:
    print(f"\nConfirmed (65-80%): {confirmed_count}/{n}")
    print(
        f"Peak depth — mean: {np.mean(peak_pcts):.1f}%  sd: {np.std(peak_pcts):.1f}%  "
        f"range: {min(peak_pcts):.0f}%–{max(peak_pcts):.0f}%"
    )

    # Depth threshold test: does peak_pct correlate with n_layers?
    n_layers_list = [all_results[m]["n_layers"] for m in all_results]
    if len(peak_pcts) >= 3:
        rho, p = scipy_stats.spearmanr(n_layers_list, peak_pcts)
        print(
            f"Depth threshold test: Spearman ρ(n_layers, peak_pct) = {rho:.3f}  p={p:.3f}"
        )
        if rho < -0.5 and p < 0.1:
            print(
                "  → Deeper models peak EARLIER as fraction: depth threshold effect confirmed"
            )
        elif abs(rho) < 0.3:
            print(
                "  → No correlation with model depth: peak % is architecture-invariant"
            )
        else:
            print("  → Weak/mixed correlation")

# ── Figure ────────────────────────────────────────────────────
family_colors = {
    "pythia": "#2E86AB",
    "gpt2": "#E84855",
    "qwen2": "#F4A261",
    "llama3": "#6A4C93",
}

n_models = len(all_results)
if n_models > 0:
    ncols = min(2, n_models)
    nrows = (n_models + ncols - 1) // ncols + 1

    fig = plt.figure(figsize=(8 * ncols, 4 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.5, wspace=0.35)

    for i, (mid, r) in enumerate(all_results.items()):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        layers = sorted(r["layer_kappas"].keys())
        kappas = [r["layer_kappas"][li] for li in layers]
        pcts = [li / r["n_layers"] * 100 for li in layers]
        color = family_colors.get(r["family"], "#888")
        ax.plot(pcts, kappas, "o-", color=color, linewidth=2, markersize=5)
        ax.axvspan(65, 80, alpha=0.12, color="green", label="Predicted 65-80%")
        ax.axvline(
            r["peak_pct"],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"Peak {r['peak_pct']:.0f}%",
        )
        q = " (8bit)" if r.get("used_8bit") else ""
        ax.set_title(
            f"{mid} {r['n_params']}{q}\n"
            f"{'✓' if r['confirmed'] else '✗'} Peak={r['peak_pct']:.0f}%  "
            f"({r['peak_layer']}/{r['n_layers']} layers)",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlabel("Depth (%)")
        ax.set_ylabel("κ̂")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Summary bar
    ax_s = fig.add_subplot(gs[-1, :])
    mids = list(all_results.keys())
    peaks = [all_results[m]["peak_pct"] for m in mids]
    bcols = [family_colors.get(all_results[m]["family"], "#888") for m in mids]
    labels = [
        f"{m}\n({all_results[m]['n_params']}, {all_results[m]['n_layers']}L)"
        for m in mids
    ]
    bars = ax_s.bar(labels, peaks, color=bcols, edgecolor="white", alpha=0.85)
    ax_s.axhspan(65, 80, alpha=0.12, color="green")
    ax_s.axhline(
        75,
        color="green",
        linestyle=":",
        linewidth=1.5,
        alpha=0.6,
        label="Prior: pythia-160m (75%)",
    )
    ax_s.set_ylim(0, 110)
    ax_s.set_ylabel("Peak κ̂ depth (%)")
    ax_s.legend(fontsize=8)
    ax_s.grid(alpha=0.3, axis="y")
    ax_s.set_title("Architecture invariance: peak curvature depth", fontweight="bold")
    for bar, val in zip(bars, peaks):
        ax_s.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.0f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        "HSG Exp 4 — Layer-Depth Sweep (AWS g5.xlarge)\nSingleton, M.",
        fontsize=12,
        fontweight="bold",
    )
    fig_path = f"{OUTPUT_DIR}/exp4_aws.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure: {fig_path}")

print("\nDone.")
