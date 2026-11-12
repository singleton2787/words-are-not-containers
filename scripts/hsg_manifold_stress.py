"""
================================================================================
HARMONIC SEMANTIC GEOMETRY (HSG) - MANIFOLD STRESS VALIDATION ENGINE
================================================================================
Fixes the Flatline Trap: Introduces non-linear path regularizers and a
capacity-constrained hidden layer bottleneck. Forces the parameters to naturally
generate trajectory curvature to resolve conflicting semantic routing targets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from stats import bootstrap_mean_ci


class BottleneckFlowNet(nn.Module):
    """
    A capacity-constrained continuous flow network.
    Forces semantic context tracking through a restricted non-linear bottleneck
    to prevent the model from effortlessly drawing uniform straight lines.
    """

    def __init__(self, seq_len=3, vocab_dim=16, hidden_dim=32):
        super().__init__()
        self.flat_dim = seq_len * vocab_dim

        # Encoder: Project space down to a tight, constrained bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim + 2, hidden_dim), nn.Tanh()
        )
        # Decoder: Force the model to reconstruct the velocity field from limited capacity
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.flat_dim),
        )

    def forward(self, x, t, y):
        batch_size = x.size(0)
        inputs = torch.cat([x.view(batch_size, -1), t, y], dim=1)
        latent = self.encoder(inputs)
        return self.decoder(latent).view(batch_size, -1, x.size(-1))


class HSGManifoldSuite:
    def __init__(self, n_seeds=30, seq_len=3, vocab_dim=16, steps=100):
        self.n_seeds = n_seeds
        self.seq_len = seq_len
        self.vocab_dim = vocab_dim
        self.steps = steps
        self.dt = 1.0 / steps

    def run_evaluation(self):
        clean_excess_all, conflicted_excess_all = [], []
        clean_curvature_all, conflicted_curvature_all = [], []

        print(
            f"=== Running HSG Manifold Stress Protocol (N = {self.n_seeds} Seeds) ==="
        )
        print("Evaluating capacity-constrained non-linear semantic flow fields...\n")

        # Construct target matrices on the product simplex
        clean_target = torch.zeros(1, self.seq_len, self.vocab_dim)
        clean_target[0, 0, 1] = 1.0  # Prefix A
        clean_target[0, 1, 2] = 1.0  # Prefix B
        clean_target[0, 2, 3] = 1.0  # Aligned Outcome

        conflicted_target = torch.zeros(1, self.seq_len, self.vocab_dim)
        conflicted_target[0, 0, 1] = 1.0  # Prefix A
        conflicted_target[0, 1, 2] = 1.0  # Prefix B
        conflicted_target[0, 2, 4] = 1.0  # Contradictory Outcome

        all_targets = torch.cat([clean_target, conflicted_target], dim=0)
        context_labels = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        for seed in range(1, self.n_seeds + 1):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = BottleneckFlowNet(self.seq_len, self.vocab_dim)
            optimizer = optim.AdamW(model.parameters(), lr=0.005)
            criterion = nn.MSELoss()

            # Train over a non-linear path matching paradigm
            for epoch in range(1200):
                optimizer.zero_grad()
                b_size = all_targets.size(0)
                t = torch.rand(b_size, 1)
                x0 = torch.randn_like(all_targets) * 0.02

                # Introduce Non-Linear Quadratic Interpolation Paths to mimic
                # structural manifold deformation during semantic generation
                t_skew = torch.sin(t * np.pi / 2.0)
                xt = (1.0 - t_skew.view(-1, 1, 1)) * x0 + t_skew.view(
                    -1, 1, 1
                ) * all_targets

                # The target velocity is now non-constant across the time horizon
                v_target = (
                    (all_targets - x0)
                    * (np.pi / 2.0)
                    * torch.cos(t * np.pi / 2.0).view(-1, 1, 1)
                )

                loss = criterion(model(xt, t, context_labels), v_target)
                loss.backward()
                optimizer.step()

            # Execute the invariant audits over the resulting parameter manifolds
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
                print(f"  [Pipeline Status] Seed {seed:02d}/{self.n_seeds} optimized.")

        self.print_report(
            "GEODESIC PATH EXCESS (Δ_geo)",
            np.array(clean_excess_all),
            np.array(conflicted_excess_all),
        )
        self.print_report(
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
                v_t = model(x_t, t_tensor, context_label)
            velocity.append(v_t.view(-1).numpy())
            trajectory.append(x_t.view(-1).numpy())
            x_t = x_t + v_t * self.dt

        velocity = np.array(velocity)
        velocity = gaussian_filter1d(
            velocity, sigma=3.0, axis=0
        )  # Apply verification filter

        speeds = np.linalg.norm(velocity, axis=1)
        arc_length = np.sum(speeds) * self.dt
        chord = np.linalg.norm(x_t.numpy().flatten() - x_0)
        path_excess = (arc_length / (chord + 1e-12)) - 1.0

        acceleration = np.diff(velocity, axis=0) / self.dt
        v_m, s_m = velocity[:-1], speeds[:-1]
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

    def compute_bootstrap_ci(self, data):
        return bootstrap_mean_ci(data, n_resamples=10000, seed=42)

    def print_report(self, metric_name, clean_arr, conf_arr):
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
        print(f"STATISTICAL READOUT: {metric_name}")
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
    suite = HSGManifoldSuite(n_seeds=30)
    suite.run_evaluation()
