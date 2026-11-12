"""
================================================================================
HARMONIC SEMANTIC GEOMETRY (HSG) - CONDITIONED MULTI-SEED SUITE
================================================================================
Fixes the Zero-Gradient Symmetry Trap by introducing a formal conditional input
variable (y) to the continuous flow architecture, breaking the averaging effect
and allowing the parameters to map separate semantic landscapes natively.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from stats import bootstrap_mean_ci


class ConditionedFlowNet(nn.Module):
    """Flow network natively conditioned on time (t) and semantic context (y)."""

    def __init__(self, seq_len=3, vocab_dim=16, d_model=64):
        super().__init__()
        self.flat_dim = seq_len * vocab_dim
        # Input features = coordinates + time (1) + context condition (1)
        self.net = nn.Sequential(
            nn.Linear(self.flat_dim + 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, self.flat_dim),
        )

    def forward(self, x, t, y):
        batch_size = x.size(0)
        inputs = torch.cat([x.view(batch_size, -1), t, y], dim=1)
        return self.net(inputs).view(batch_size, -1, x.size(-1))


class HSGConditionedSuite:
    def __init__(self, n_seeds=30, seq_len=3, vocab_dim=16, steps=100):
        self.n_seeds = n_seeds
        self.seq_len = seq_len
        self.vocab_dim = vocab_dim
        self.steps = steps
        self.dt = 1.0 / steps

    def run_full_evaluation(self):
        clean_excess_all = []
        conflicted_excess_all = []
        clean_curvature_all = []
        conflicted_curvature_all = []

        print(
            f"=== Starting Conditioned Multi-Seed Protocol (N = {self.n_seeds} Seeds) ==="
        )
        print("Breaking symmetry via conditional context routing vectors...\n")

        # Define targets on the simplex space
        clean_target = torch.zeros(1, self.seq_len, self.vocab_dim)
        clean_target[0, 0, 1] = 1.0  # Prefix token 1
        clean_target[0, 1, 2] = 1.0  # Prefix token 2
        clean_target[0, 2, 3] = 1.0  # Aligned outcome

        conflicted_target = torch.zeros(1, self.seq_len, self.vocab_dim)
        conflicted_target[0, 0, 1] = 1.0  # Prefix token 1
        conflicted_target[0, 1, 2] = 1.0  # Prefix token 2
        conflicted_target[0, 2, 4] = 1.0  # Contradictory outcome

        all_targets = torch.cat([clean_target, conflicted_target], dim=0)

        # Explicit context tokens: 0.0 for clean path, 1.0 for conflicted path
        context_labels = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        for seed in range(1, self.n_seeds + 1):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = ConditionedFlowNet(self.seq_len, self.vocab_dim)
            optimizer = optim.AdamW(model.parameters(), lr=0.005)
            criterion = nn.MSELoss()

            # Train conditional trajectories until loss convergence
            for epoch in range(1000):
                optimizer.zero_grad()
                b_size = all_targets.size(0)
                t = torch.rand(b_size, 1)
                x0 = torch.randn_like(all_targets) * 0.02

                # Sincere optimal transport displacement maps
                xt = (1.0 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * all_targets
                v_target = all_targets - x0

                # Pass the context label tensor directly into the network
                loss = criterion(model(xt, t, context_labels), v_target)
                loss.backward()
                optimizer.step()

            # Audit both conditions separately using their respective context embeddings
            c_excess, c_curv = self.integrate_path(
                model, clean_target, torch.tensor([[0.0]]), seed
            )
            x_excess, x_curv = self.integrate_path(
                model, conflicted_target, torch.tensor([[1.0]]), seed
            )

            clean_excess_all.append(c_excess)
            conflicted_excess_all.append(x_excess)
            clean_curvature_all.append(c_curv)
            conflicted_curvature_all.append(x_curv)

            if seed % 5 == 0 or seed == 1:
                print(f"  [Progress Log] Seed {seed:02d}/{self.n_seeds} complete.")

        self.print_statistical_report(
            "GEODESIC PATH EXCESS (Δ_geo)",
            np.array(clean_excess_all),
            np.array(conflicted_excess_all),
        )
        self.print_statistical_report(
            "TOTAL INTEGRATED ARC LENGTH CURVATURE (K_total)",
            np.array(clean_curvature_all),
            np.array(conflicted_curvature_all),
        )

    def integrate_path(self, model, target_instance, context_label, seed):
        model.eval()
        torch.manual_seed(seed + 1000)
        x_t = torch.randn_like(target_instance) * 0.02
        x_0 = x_t.clone().numpy().flatten()

        trajectory, velocity = [], []
        for step in range(self.steps):
            t_val = step * self.dt
            t_tensor = torch.tensor([[t_val]], dtype=torch.float32)
            with torch.no_grad():
                # Enforce the specific model conditioning channel step
                v_t = model(x_t, t_tensor, context_label)
            velocity.append(v_t.view(-1).numpy())
            trajectory.append(x_t.view(-1).numpy())
            x_t = x_t + v_t * self.dt

        velocity = np.array(velocity)
        velocity = gaussian_filter1d(velocity, sigma=3.0, axis=0)

        speeds = np.linalg.norm(velocity, axis=1)
        arc_length = np.sum(speeds) * self.dt
        chord = np.linalg.norm(x_t.numpy().flatten() - x_0)
        path_excess = (arc_length / (chord + 1e-12)) - 1.0

        acceleration = np.diff(velocity, axis=0) / self.dt
        v_m = velocity[:-1]
        s_m = speeds[:-1]
        k_total = 0.0
        for t in range(len(acceleration)):
            v_n = s_m[t]
            if v_n < 1e-5:
                continue
            num = np.sqrt(
                max(
                    0.0,
                    (v_n**2) * (np.linalg.norm(acceleration[t]) ** 2)
                    - (np.dot(v_m[t], acceleration[t]) ** 2),
                )
            )
            k_total += (num / (v_n**3)) * (v_n * self.dt)

        return path_excess, k_total

    def compute_bootstrap_ci(self, data, n_resamples=10000, ci=0.95):
        return bootstrap_mean_ci(data, n_resamples=n_resamples, ci=ci, seed=42)

    def print_statistical_report(self, metric_name, clean_arr, conf_arr):
        m_clean, m_conf = np.mean(clean_arr), np.mean(conf_arr)
        std_clean, std_conf = np.std(clean_arr, ddof=1), np.std(conf_arr, ddof=1)
        ci_clean_low, ci_clean_high = self.compute_bootstrap_ci(clean_arr)
        ci_conf_low, ci_conf_high = self.compute_bootstrap_ci(conf_arr)

        t_stat, p_val = stats.ttest_ind(clean_arr, conf_arr, equal_var=False)
        pooled_std = np.sqrt(
            ((len(clean_arr) - 1) * std_clean**2 + (len(conf_arr) - 1) * std_conf**2)
            / (len(clean_arr) + len(conf_arr) - 2)
        )
        cohen_d = (m_conf - m_clean) / (pooled_std + 1e-12)

        print("\n" + "=" * 80)
        print(f"METRIC EVALUATION FLUX: {metric_name}")
        print("=" * 80)
        print(f"Mean Clean Environment        -> {m_clean:.5f}")
        print(
            f"  95% Empirical Bootstrap CI  -> [{ci_clean_low:.5f}, {ci_clean_high:.5f}]"
        )
        print(f"Mean Conflicted Environment   -> {m_conf:.5f}")
        print(
            f"  95% Empirical Bootstrap CI  -> [{ci_conf_low:.5f}, {ci_conf_high:.5f}]"
        )
        print("-" * 80)
        print(
            f"Net Separation Scale Ratio    -> {m_conf / (m_clean + 1e-12):.2f}x separation"
        )
        print(f"Welch's t-test p-value        -> {p_val:.5e}")
        print(f"Standardized Effect Size (d)  -> {cohen_d:.3f} Standard Deviations")
        print("=" * 80)


if __name__ == "__main__":
    suite = HSGConditionedSuite(n_seeds=30)
    suite.run_full_evaluation()
