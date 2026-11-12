"""
hsg_probe.py — Harmonic Semantic Geometry Probe
================================================
Singleton, M.
v0.1 — June 2026

Single-file implementation. No dependencies beyond transformers + torch + scipy.

Usage:
    from hsg_probe import HSGProbe
    probe = HSGProbe("Qwen/Qwen2-0.5B")
    result = probe.evaluate("Look, I love my wife, okay?")
    print(result)
    # {'kappa': 0.847, 'kappa_pct': 78.3, 'tau': 0.021, 'flag': True,
    #  'layer': 22, 'n_tokens': 12, 'model': 'Qwen/Qwen2-0.5B'}

Ablation suite:
    python hsg_probe.py --ablate --model Qwen/Qwen2-0.5B

Full test suite (send to Nelson):
    python hsg_probe.py --test --model Qwen/Qwen2-0.5B
"""

import json
import warnings
import hashlib
from typing import Optional
import numpy as np

warnings.filterwarnings("ignore")


# ── Geometry ─────────────────────────────────────────────────────────────────


def arc_length_reparameterize(traj: np.ndarray, n_out: int = 100):
    """Re-parameterize trajectory by arc length. Returns (traj_s, total_length)."""
    if len(traj) < 2:
        return traj, 0.0
    diffs = np.diff(traj, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-10:
        return traj, 0.0
    s = np.linspace(0, total, n_out)
    ts = np.column_stack([np.interp(s, cum, traj[:, d]) for d in range(traj.shape[1])])
    return ts, total


def curvature_hd(ts: np.ndarray) -> np.ndarray:
    """
    Arc-length-parameterized curvature in R^d.
    κ(s) = |component of γ''(s) orthogonal to γ'(s)|
    """
    d1 = np.gradient(ts, axis=0)
    d2 = np.gradient(d1, axis=0)
    n = np.linalg.norm(d1, axis=1, keepdims=True)
    u = d1 / np.where(n > 1e-10, n, 1.0)
    kv = d2 - np.einsum("ij,ij->i", d2, u)[:, None] * u
    return np.linalg.norm(kv, axis=1)


def torsion_hd(ts: np.ndarray) -> np.ndarray:
    """
    Generalized torsion projected onto first 3 Frenet dimensions.
    τ(s) = [(γ'×γ'')·γ'''] / |γ'×γ''|²
    """
    d1 = np.gradient(ts, axis=0)
    d2 = np.gradient(d1, axis=0)
    d3 = np.gradient(d2, axis=0)
    tau = np.zeros(len(ts))
    for i in range(1, len(ts) - 1):
        c = np.cross(d1[i, :3], d2[i, :3])
        den = np.dot(c, c)
        if den > 1e-10:
            tau[i] = np.dot(c, d3[i, :3]) / den
    return tau


def excess_arc_length(ts: np.ndarray, total: float) -> float:
    """Δ = L(γ)/||γ(1)-γ(0)|| - 1. Zero for straight-line trajectory."""
    chord = np.linalg.norm(ts[-1] - ts[0])
    return (total / chord - 1.0) if chord > 1e-10 else 0.0


def compute_geometry(traj: np.ndarray, n_resample: int = 100) -> dict:
    """Full geometric characterization of a residual-stream trajectory."""
    if len(traj) < 4:
        return {
            "kappa_mean": 0.0,
            "kappa_max": 0.0,
            "tau_abs_mean": 0.0,
            "delta": 0.0,
            "n_tokens": len(traj),
        }
    ts, L = arc_length_reparameterize(traj, n_resample)
    kappa = curvature_hd(ts)
    tau = torsion_hd(ts)
    delta = excess_arc_length(ts, L)
    n_tok = len(traj)
    # Length-normalized curvature: κ̂ / sqrt(n_tokens)
    # Corrects for trajectory-length-dependent curvature accumulation.
    # sqrt rather than linear because curvature accumulates sub-linearly
    # with sequence length (empirically validated in ablation 3).
    kappa_norm = float(kappa.mean()) / np.sqrt(max(n_tok, 1))
    return {
        "kappa_mean": float(kappa.mean()),
        "kappa_norm": kappa_norm,
        "kappa_max": float(kappa.max()),
        "tau_abs_mean": float(np.abs(tau).mean()),
        "tau_signed_mean": float(tau.mean()),
        "delta": float(delta),
        "n_tokens": n_tok,
        "arc_length": float(L),
    }


# ── Normalization ─────────────────────────────────────────────────────────────


class BaselineDistribution:
    """
    Per-model baseline κ̂ distribution, built from a reference corpus.
    Needed to convert raw κ̂ to a percentile score.
    Addresses Nelson's hole #2 and #4.
    """

    REFERENCE_PROMPTS = [
        # Factual — expected low κ̂
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "The Earth orbits the Sun.",
        "Shakespeare wrote Hamlet.",
        "The speed of light is approximately 300,000 km/s.",
        "Mount Everest is the tallest mountain on Earth.",
        # Sincere — expected low-medium κ̂
        "I love my wife, and I always have.",
        "This is excellent work.",
        "I am genuinely impressed by this.",
        "She means everything to me.",
        # Complex — expected higher κ̂
        "Look, I love my wife, okay?",
        "Oh great, another Monday.",
        "The presentation was adequate.",
        "After years of silence, she finally called.",
        "Some people believe capital punishment is justified.",
        "It is sometimes said that ends justify means.",
        # Irony/defensiveness — expected high κ̂
        "Wonderful. The printer is broken again.",
        "Just what I needed: more meetings.",
        "Why would you even question that? I love my wife.",
        "I have always loved my wife, whatever you think.",
    ]

    def __init__(self):
        self.kappas = []
        self.fitted = False
        self.mean = 0.0
        self.std = 1.0
        self.p95 = 1.0

    def fit(self, probe: "HSGProbe", verbose: bool = False) -> "BaselineDistribution":
        """Build baseline from reference corpus. Runs ~20 forward passes."""
        print(
            f"  Building baseline distribution ({len(self.REFERENCE_PROMPTS)} prompts)..."
        )
        for p in self.REFERENCE_PROMPTS:
            g = probe._extract_and_compute(p, probe.peak_layer)
            if g:
                self.kappas.append(g["kappa_mean"])
                if verbose:
                    print(f"    {p[:50]:<50} κ̂={g['kappa_mean']:.4f}")

        self.kappas = np.array(self.kappas)
        self.mean = float(self.kappas.mean())
        self.std = float(self.kappas.std())
        self.p95 = float(np.percentile(self.kappas, 95))
        self.fitted = True
        print(
            f"  Baseline: mean={self.mean:.4f}  std={self.std:.4f}  p95={self.p95:.4f}"
        )
        return self

    def percentile(self, kappa: float) -> float:
        """Return percentile of kappa in baseline distribution."""
        if not self.fitted or len(self.kappas) == 0:
            return 50.0
        return float((self.kappas < kappa).mean() * 100)

    def z_score(self, kappa: float) -> float:
        """Z-score relative to baseline."""
        if self.std < 1e-10:
            return 0.0
        return (kappa - self.mean) / self.std


# ── Core probe ────────────────────────────────────────────────────────────────


class HSGProbe:
    """
    Harmonic Semantic Geometry Probe.

    Extracts residual-stream trajectories at the peak curvature layer
    (n_layers - 2, the structural invariant from Experiment 4) and
    computes arc-length-parameterized curvature κ̂ and torsion τ.

    Parameters
    ----------
    model_name : HuggingFace model identifier
    layer_offset : layers from end (default=2, from Experiment 4 invariant)
    cache_dir : HuggingFace cache directory
    device : 'cuda', 'cpu', or 'auto'
    flag_threshold_pct : percentile above which a prompt is flagged (default=75)
    """

    def __init__(
        self,
        model_name: str,
        layer_offset: int = 2,
        cache_dir: Optional[str] = None,
        device: str = "auto",
        flag_threshold_pct: float = 75.0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.layer_offset = layer_offset
        self.flag_threshold_pct = flag_threshold_pct
        self.baseline = None
        self._cache = {}  # reproducibility cache

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            output_hidden_states=True,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size
        self.peak_layer = self.n_layers - self.layer_offset

        print(
            f"  {self.n_layers}L × d{self.d_model}  "
            f"peak_layer={self.peak_layer} (n_layers - {self.layer_offset})"
        )

    def build_baseline(self, verbose: bool = False) -> "HSGProbe":
        """Build baseline distribution for percentile scoring."""
        self.baseline = BaselineDistribution().fit(self, verbose)
        return self

    def _extract_trajectory(
        self, prompt: str, layer_idx: int, max_length: int = 64
    ) -> Optional[np.ndarray]:
        """Extract hidden states at layer_idx for prompt. Returns (T, d) array."""
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )
        dev = next(self.model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states  # tuple: (n_layers+1) × (1, T, d)
        idx = layer_idx % len(hs)
        return hs[idx][0].float().cpu().numpy()

    def _extract_and_compute(
        self, prompt: str, layer_idx: int, n_resample: int = 100
    ) -> Optional[dict]:
        """Extract trajectory and compute full geometry."""
        traj = self._extract_trajectory(prompt, layer_idx)
        if traj is None or len(traj) < 4:
            return None
        g = compute_geometry(traj, n_resample)
        return g

    def evaluate(
        self,
        prompt: str,
        layer_offset: Optional[int] = None,
        n_resample: int = 100,
        use_cache: bool = True,
    ) -> dict:
        """
        Evaluate a single prompt.

        Returns
        -------
        dict with keys:
            kappa       : mean arc-length-parameterized curvature (raw)
            kappa_pct   : percentile in baseline distribution (0-100)
            kappa_z     : z-score relative to baseline
            tau         : mean |torsion| (double entendre index)
            delta       : excess arc length (communicative indirection)
            flag        : True if kappa_pct > flag_threshold_pct
            layer       : layer index used
            n_tokens    : number of tokens in prompt
            model       : model name
        """
        layer = self.n_layers - (layer_offset or self.layer_offset)

        # Reproducibility cache
        cache_key = hashlib.md5(
            f"{prompt}|{layer}|{self.model_name}".encode()
        ).hexdigest()
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        g = self._extract_and_compute(prompt, layer, n_resample)
        if g is None:
            return {"error": "insufficient tokens", "prompt": prompt}

        kappa_pct = self.baseline.percentile(g["kappa_mean"]) if self.baseline else None
        kappa_z = self.baseline.z_score(g["kappa_mean"]) if self.baseline else None

        result = {
            "kappa": round(g["kappa_mean"], 5),
            "kappa_norm": round(g["kappa_norm"], 5),
            "kappa_max": round(g["kappa_max"], 5),
            "kappa_pct": round(kappa_pct, 1) if kappa_pct is not None else None,
            "kappa_z": round(kappa_z, 3) if kappa_z is not None else None,
            "tau": round(g["tau_abs_mean"], 5),
            "tau_signed": round(g["tau_signed_mean"], 5),
            "delta": round(g["delta"], 5),
            "flag": (kappa_pct > self.flag_threshold_pct)
            if kappa_pct is not None
            else None,
            "layer": layer,
            "n_tokens": g["n_tokens"],
            "model": self.model_name,
        }

        if use_cache:
            self._cache[cache_key] = result
        return result

    def evaluate_batch(self, prompts: list, **kwargs) -> list:
        """Evaluate a list of prompts. Returns list of result dicts."""
        return [self.evaluate(p, **kwargs) for p in prompts]


# ── Ablation suite ────────────────────────────────────────────────────────────


class HSGAblationSuite:
    """
    Five ablation tests addressing Nelson's anticipated holes.

    Ablation 1 — Layer sensitivity:
        Does the flag change if we use n_layers-3 instead of n_layers-2?
        Acceptable: <20% of prompts flip. Unacceptable: >40% flip.

    Ablation 2 — Within-model normalization:
        Are raw κ̂ scores comparable across prompt categories after
        baseline normalization? (Addresses scaling confound.)

    Ablation 3 — Length confound:
        Does κ̂ correlate with prompt length (n_tokens)?
        Acceptable: |r| < 0.3. Unacceptable: |r| > 0.5.

    Ablation 4 — Reproducibility:
        Same prompt, same layer, 5 consecutive runs.
        Acceptable: CV < 1%. Unacceptable: CV > 5%.

    Ablation 5 — Category separation:
        Do high-pragmatic-complexity categories show higher κ̂ than
        low-complexity categories? Mann-Whitney U test.
    """

    HIGH_COMPLEXITY = [
        "Look, I love my wife, okay?",
        "Oh great, another Monday.",
        "Wonderful. The printer is broken again.",
        "I said I love my wife and I mean it.",
        "Why would you even question that? I love my wife.",
        "Just what I needed: more meetings.",
        "The presentation was adequate.",
        "The food was fine, I suppose.",
        "After years of silence, she finally called.",
        "He left without saying goodbye.",
    ]

    LOW_COMPLEXITY = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "The Earth orbits the Sun.",
        "Shakespeare wrote Hamlet.",
        "This is excellent work.",
        "The results are outstanding.",
        "I am genuinely impressed.",
        "I love my wife, and I always have.",
        "She means everything to me.",
        "The speed of light is approximately 300,000 km/s.",
    ]

    def __init__(self, probe: HSGProbe):
        self.probe = probe
        self.results = {}

    def run_all(self, verbose: bool = True) -> dict:
        """Run all five ablations. Returns results dict."""
        print(f"\n{'═' * 60}")
        print(f"HSG ABLATION SUITE — {self.probe.model_name}")
        print(
            f"Peak layer: {self.probe.peak_layer}  "
            f"(n_layers={self.probe.n_layers}, offset={self.probe.layer_offset})"
        )
        print(f"{'═' * 60}")

        self.results["ablation_1"] = self._ablation_1_layer_sensitivity(verbose)
        self.results["ablation_2"] = self._ablation_2_normalization(verbose)
        self.results["ablation_3"] = self._ablation_3_length_confound(verbose)
        self.results["ablation_4"] = self._ablation_4_reproducibility(verbose)
        self.results["ablation_5"] = self._ablation_5_category_separation(verbose)

        self._print_summary()
        return self.results

    def _ablation_1_layer_sensitivity(self, verbose: bool) -> dict:
        """Test: does score change if we use offset-1 or offset+1?"""
        print("\n--- Ablation 1: Layer sensitivity ---")
        all_prompts = self.HIGH_COMPLEXITY + self.LOW_COMPLEXITY
        offset_default = self.probe.layer_offset

        scores = {}
        for offset in [offset_default - 1, offset_default, offset_default + 1]:
            if offset < 1 or offset > self.probe.n_layers - 1:
                continue
            scores[offset] = []
            for p in all_prompts:
                r = self.probe.evaluate(p, layer_offset=offset, use_cache=False)
                scores[offset].append(r.get("kappa", 0.0))

        # Compute flag flip rate between default and adjacent layers
        flip_rates = {}
        default_scores = np.array(scores.get(offset_default, []))
        for offset, sc in scores.items():
            if offset == offset_default:
                continue
            sc_arr = np.array(sc)
            # Use p75 of default as threshold
            thresh = np.percentile(default_scores, 75)
            flags_default = default_scores > thresh
            flags_other = sc_arr > thresh
            flip_rate = float((flags_default != flags_other).mean() * 100)
            flip_rates[f"offset_{offset}_vs_{offset_default}"] = flip_rate
            if verbose:
                corr = float(np.corrcoef(default_scores, sc_arr)[0, 1])
                print(
                    f"  Offset {offset} vs {offset_default}: "
                    f"flip_rate={flip_rate:.1f}%  r={corr:.3f}"
                )

        max_flip = max(flip_rates.values()) if flip_rates else 0
        passed = max_flip < 20
        print(
            f"  Max flip rate: {max_flip:.1f}%  → {'PASS (<20%)' if passed else 'FAIL (>20%)'}"
        )
        return {"flip_rates": flip_rates, "max_flip_pct": max_flip, "passed": passed}

    def _ablation_2_normalization(self, verbose: bool) -> dict:
        """Test: after baseline normalization, do percentiles separate categories?"""
        print("\n--- Ablation 2: Normalization ---")
        if self.probe.baseline is None:
            print("  No baseline fitted — building now...")
            self.probe.build_baseline()

        high_pcts, low_pcts = [], []
        for p in self.HIGH_COMPLEXITY:
            r = self.probe.evaluate(p)
            if r.get("kappa_pct") is not None:
                high_pcts.append(r["kappa_pct"])
        for p in self.LOW_COMPLEXITY:
            r = self.probe.evaluate(p)
            if r.get("kappa_pct") is not None:
                low_pcts.append(r["kappa_pct"])

        high_mean = np.mean(high_pcts)
        low_mean = np.mean(low_pcts)
        separation = high_mean - low_mean

        if verbose:
            print(f"  High complexity: mean pct={high_mean:.1f}%  (n={len(high_pcts)})")
            print(f"  Low complexity:  mean pct={low_mean:.1f}%  (n={len(low_pcts)})")
            print(f"  Separation: {separation:.1f} percentile points")

        passed = separation > 10
        print(f"  → {'PASS (>10 pct pts)' if passed else 'FAIL (<10 pct pts)'}")
        return {
            "high_mean_pct": high_mean,
            "low_mean_pct": low_mean,
            "separation_pct": separation,
            "passed": passed,
        }

    def _ablation_3_length_confound(self, verbose: bool) -> dict:
        """
        Test: is kappa_norm (length-corrected) still correlated with length?
        Also tests raw kappa for comparison.
        Uses length-matched pairs to avoid corpus-level confound.
        """
        from scipy import stats as ss

        print("\n--- Ablation 3: Length confound ---")
        all_prompts = self.HIGH_COMPLEXITY + self.LOW_COMPLEXITY
        kappas_raw, kappas_norm, lengths = [], [], []
        for p in all_prompts:
            g = self.probe._extract_and_compute(p, self.probe.peak_layer)
            if g:
                kappas_raw.append(g["kappa_mean"])
                kappas_norm.append(g["kappa_norm"])
                lengths.append(g["n_tokens"])

        r_raw, p_raw = ss.pearsonr(lengths, kappas_raw)
        r_norm, p_norm = ss.pearsonr(lengths, kappas_norm)

        if verbose:
            print(f"  Raw  κ̂:  Pearson r(length, κ̂)      = {r_raw:.3f}  p={p_raw:.4f}")
            print(
                f"  Norm κ̂:  Pearson r(length, κ̂_norm) = {r_norm:.3f}  p={p_norm:.4f}"
            )
            print(f"  Length range: {min(lengths)}-{max(lengths)} tokens")
            print("  Length-matched test (same n_tokens, different content):")

            # Length-matched spot check
            matched_pairs = [
                ("I love my wife.", "This is excellent work."),
                ("Oh great, another Monday.", "The results are outstanding."),
            ]
            for p1, p2 in matched_pairs:
                g1 = self.probe._extract_and_compute(p1, self.probe.peak_layer)
                g2 = self.probe._extract_and_compute(p2, self.probe.peak_layer)
                if g1 and g2:
                    print(
                        f"    '{p1}' ({g1['n_tokens']}t) κ̂_norm={g1['kappa_norm']:.4f}"
                    )
                    print(
                        f"    '{p2}' ({g2['n_tokens']}t) κ̂_norm={g2['kappa_norm']:.4f}"
                    )

        passed = abs(r_norm) < 0.3
        improvement = abs(r_raw) - abs(r_norm)
        print(
            f"  Raw r={r_raw:.3f} → Normalized r={r_norm:.3f} (improvement: {improvement:.3f})"
        )
        print(
            f"  → {'PASS (|r_norm|<0.3)' if passed else 'WARN — residual length effect after normalization'}"
        )
        return {
            "pearson_r_raw": r_raw,
            "pearson_r_norm": r_norm,
            "p_value": p_norm,
            "improvement": improvement,
            "passed": passed,
        }

    def _ablation_4_reproducibility(self, verbose: bool) -> dict:
        """Test: same prompt 5× — is κ̂ stable?"""
        print("\n--- Ablation 4: Reproducibility ---")
        test_prompts = [
            "Oh great, another Monday.",
            "The capital of France is Paris.",
            "Look, I love my wife, okay?",
        ]
        cvs = []
        for p in test_prompts:
            runs = []
            for _ in range(5):
                r = self.probe.evaluate(p, use_cache=False)
                runs.append(r.get("kappa", 0.0))
            mean_k = np.mean(runs)
            std_k = np.std(runs)
            cv = (std_k / mean_k * 100) if mean_k > 1e-10 else 0.0
            cvs.append(cv)
            if verbose:
                print(
                    f"  '{p[:40]}...'  mean={mean_k:.5f}  std={std_k:.5f}  CV={cv:.2f}%"
                )

        max_cv = max(cvs)
        passed = max_cv < 1.0
        print(f"  Max CV: {max_cv:.2f}%  → {'PASS (<1%)' if passed else 'WARN (>1%)'}")
        return {"cvs": cvs, "max_cv": max_cv, "passed": passed}

    def _ablation_5_category_separation(self, verbose: bool) -> dict:
        """Test: Mann-Whitney U — do high > low complexity prompts show higher κ̂?"""
        from scipy import stats as ss

        print("\n--- Ablation 5: Category separation ---")
        # Use kappa_norm (length-corrected) for fair comparison
        high_k = [
            self.probe._extract_and_compute(p, self.probe.peak_layer)["kappa_norm"]
            for p in self.HIGH_COMPLEXITY
            if self.probe._extract_and_compute(p, self.probe.peak_layer)
        ]
        low_k = [
            self.probe._extract_and_compute(p, self.probe.peak_layer)["kappa_norm"]
            for p in self.LOW_COMPLEXITY
            if self.probe._extract_and_compute(p, self.probe.peak_layer)
        ]

        u_stat, p_val = ss.mannwhitneyu(high_k, low_k, alternative="greater")
        d = (np.mean(high_k) - np.mean(low_k)) / np.sqrt(
            (np.std(high_k) ** 2 + np.std(low_k) ** 2) / 2
        )

        if verbose:
            print(f"  High: mean κ̂={np.mean(high_k):.4f}  (n={len(high_k)})")
            print(f"  Low:  mean κ̂={np.mean(low_k):.4f}  (n={len(low_k)})")
            print(f"  Mann-Whitney U={u_stat:.0f}  p={p_val:.4f}  Cohen's d={d:.3f}")

        passed = p_val < 0.05 and d > 0
        verdict = (
            "SIGNIFICANT"
            if p_val < 0.05 and d > 0
            else "DIRECTIONAL"
            if d > 0
            else "NO SIGNAL"
        )
        print(f"  → {verdict}")
        return {
            "u_stat": u_stat,
            "p_value": p_val,
            "cohens_d": d,
            "passed": passed,
            "verdict": verdict,
        }

    def _print_summary(self):
        print(f"\n{'═' * 60}")
        print("ABLATION SUMMARY")
        print(f"{'═' * 60}")
        tests = [
            ("Ablation 1", "Layer sensitivity", "ablation_1"),
            ("Ablation 2", "Normalization", "ablation_2"),
            ("Ablation 3", "Length confound", "ablation_3"),
            ("Ablation 4", "Reproducibility", "ablation_4"),
            ("Ablation 5", "Category separation", "ablation_5"),
        ]
        all_pass = True
        for label, name, key in tests:
            r = self.results.get(key, {})
            passed = r.get("passed", False)
            status = "PASS" if passed else "FAIL/WARN"
            if not passed:
                all_pass = False
            print(f"  {label}: {name:<25} {status}")
        print(
            f"\n  Overall: {'ALL PASS — ready to send to Nelson' if all_pass else 'ISSUES — fix before sending'}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HSG Probe")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--offset", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    probe = HSGProbe(args.model, layer_offset=args.offset)

    if args.baseline or args.ablate or args.test:
        probe.build_baseline(verbose=args.baseline)

    if args.prompt:
        result = probe.evaluate(args.prompt)
        print(json.dumps(result, indent=2))

    if args.ablate or args.test:
        suite = HSGAblationSuite(probe)
        results = suite.run_all(verbose=True)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=float)
            print(f"\nResults saved: {args.output}")

    if not any([args.prompt, args.ablate, args.test, args.baseline]):
        # Interactive demo
        probe.build_baseline()
        demo_prompts = [
            "The capital of France is Paris.",
            "I love my wife, and I always have.",
            "The presentation was adequate.",
            "Oh great, another Monday.",
            "Look, I love my wife, okay?",
            "After years of silence, she finally called.",
        ]
        print(f"\n{'Prompt':<50} {'κ̂ raw':>8} {'κ̂ norm':>8} {'pct':>6} {'flag':>6}")
        print("-" * 82)
        for p in demo_prompts:
            r = probe.evaluate(p)
            flag = "🚩" if r.get("flag") else "  "
            pct = f"{r['kappa_pct']:.0f}%" if r.get("kappa_pct") is not None else "  -"
            norm = r.get("kappa_norm", r["kappa"])
            print(f"{p:<50} {r['kappa']:>8.4f} {norm:>8.4f} {pct:>6} {flag}")


if __name__ == "__main__":
    main()
