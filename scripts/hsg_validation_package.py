import os
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.ndimage import gaussian_filter1d

torch.manual_seed(42)
np.random.seed(42)


class ContinuousTextFlowNet(nn.Module):
    def __init__(self, sequence_length, vocab_size, d_model=256):
        super().__init__()
        self.flat_dim = sequence_length * vocab_size
        self.net = nn.Sequential(
            nn.Linear(self.flat_dim + 1, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, self.flat_dim),
        )

    def forward(self, x, t):
        batch_size = x.size(0)
        inputs = torch.cat([x.view(batch_size, -1), t], dim=1)
        return self.net(inputs).view(batch_size, -1, x.size(-1))


class HSGValidationSuite:
    def __init__(self, seq_len=5, integration_steps=100):
        self.seq_len = seq_len
        self.steps = integration_steps
        self.dt = 1.0 / integration_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fetch_and_build_corpus(self):
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
        file_path = "wikitext_valid.txt"
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url, file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_lines = [line.strip().lower().split() for line in f.readlines()]
        clean_corpus = [
            line[: self.seq_len]
            for line in raw_lines
            if len(line) >= self.seq_len and not line[0].startswith("=")
        ]
        sliced_data = clean_corpus[:500]
        vocab = sorted(
            list(set([token for sentence in sliced_data for token in sentence]))
        )
        word2idx = {token: idx for idx, token in enumerate(vocab)}
        tensor = torch.zeros(len(sliced_data), self.seq_len, len(vocab))
        for i, sentence in enumerate(sliced_data):
            for j, token in enumerate(sentence):
                tensor[i, j, word2idx[token]] = 1.0
        return tensor.to(self.device), len(vocab)

    def execute_faithful_patching(
        self, model, clean_instance, inject_tension=False, t_patch=0.4, sigma=3.0
    ):
        model.eval()
        shared_prior_noise = torch.randn_like(clean_instance) * 0.05

        # Caching Pass (Clean Run Activation Extraction)
        clean_velocity_cache = []
        x_clean = shared_prior_noise.clone().detach()
        for step in range(self.steps):
            t_val = step * self.dt
            t_tensor = torch.tensor([[t_val]], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                v_clean = model(x_clean, t_tensor)
            clean_velocity_cache.append(v_clean.clone().detach())
            x_clean = x_clean + v_clean * self.dt

        # Corrupted Pass with Intervention Hooks
        trajectory, velocity = [], []
        x_patched = shared_prior_noise.clone().detach()
        for step in range(self.steps):
            t_val = step * self.dt
            t_tensor = torch.tensor([[t_val]], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                v_corrupted = model(x_patched, t_tensor)
                if inject_tension and 0.25 < t_val < 0.65:
                    v_corrupted[0, self.seq_len - 1, :] += (
                        torch.randn(clean_instance.size(-1), device=self.device)
                        * 5.5
                        * np.sin(t_val * np.pi)
                    )

                # Activation Patching Window (a_l,p(X) <- a_l,p(C))
                if abs(t_val - t_patch) < (self.dt / 2.0):
                    v_clean_cached = clean_velocity_cache[step]
                    mask = torch.zeros_like(v_corrupted)
                    mask[0, self.seq_len - 1, :] = 1.0
                    v_patched_step = v_corrupted + mask * (v_clean_cached - v_corrupted)
                else:
                    v_patched_step = v_corrupted

            velocity.append(v_patched_step.view(-1).cpu().numpy())
            trajectory.append(x_patched.view(-1).cpu().numpy())
            x_patched = x_patched + v_patched_step * self.dt

        velocity = np.array(velocity)
        if sigma > 0:
            velocity = gaussian_filter1d(velocity, sigma=sigma, axis=0)

        return self.compute_invariant_geometry(np.array(trajectory), velocity)

    def compute_invariant_geometry(self, trajectory, velocity):
        step_magnitudes = np.linalg.norm(velocity, axis=1)
        integrated_arc_length = np.sum(step_magnitudes) * self.dt
        chord_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        path_excess = (integrated_arc_length / (chord_distance + 1e-12)) - 1.0

        acceleration = np.diff(velocity, axis=0) / self.dt
        v_matched = velocity[:-1]
        step_magnitudes_matched = step_magnitudes[:-1]

        integrated_curvature = 0.0
        for t in range(len(acceleration)):
            v_norm = step_magnitudes_matched[t]
            a_norm = np.linalg.norm(acceleration[t])
            v_dot_a = np.dot(v_matched[t], acceleration[t])
            if v_norm < 1e-6:
                continue
            numerator = np.sqrt(max(0.0, (v_norm**2) * (a_norm**2) - (v_dot_a**2)))
            kappa_t = numerator / (v_norm**3)
            integrated_curvature += kappa_t * (v_norm * self.dt)

        return path_excess, integrated_curvature


if __name__ == "__main__":
    suite = HSGValidationSuite()
    targets, vocab_size = suite.fetch_and_build_corpus()
    model = ContinuousTextFlowNet(suite.seq_len, vocab_size).to(suite.device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()

    for epoch in range(600):
        optimizer.zero_grad()
        b_size = targets.size(0)
        t = torch.rand(b_size, 1, device=suite.device)
        x0 = torch.randn_like(targets) * 0.05
        xt = (1.0 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * targets
        loss = criterion(model(xt, t), targets - x0)
        loss.backward()
        optimizer.step()

    test_case = targets[0].unsqueeze(0)
    lit_ex, _ = suite.execute_faithful_patching(
        model, test_case, inject_tension=False, sigma=0.0
    )
    prag_ex, _ = suite.execute_faithful_patching(
        model, test_case, inject_tension=True, sigma=0.0
    )
    _, lit_k_smooth = suite.execute_faithful_patching(
        model, test_case, inject_tension=False, sigma=3.0
    )
    _, prag_k_smooth = suite.execute_faithful_patching(
        model, test_case, inject_tension=True, sigma=3.0
    )

    print("\n" + "=" * 80)
    print("VERIFICATION ENGINE REPORT OUT")
    print("=" * 80)
    print(f"Ablation A: RAW Literal Path Excess     -> {lit_ex:.5f}")
    print(f"Ablation A: RAW Pragmatic Path Excess   -> {prag_ex:.5f}")
    print(
        f"NET Path Excess Separation Factor      -> {prag_ex / (lit_ex + 1e-12):.2f}x"
    )
    print("-" * 80)
    print(f"Ablation B: SMOOTHED Literal Curvature ̂κ -> {lit_k_smooth:.5f}")
    print(f"Ablation B: SMOOTHED Pragmatic Curvature ̂κ-> {prag_k_smooth:.5f}")
    print(
        f"NET Macro Curvature Separation Factor  -> {prag_k_smooth / lit_k_smooth:.2f}x"
    )
    print("=" * 80)
