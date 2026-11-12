"""
================================================================================
HARMONIC SEMANTIC GEOMETRY (HSG) - 30-SEED STATISTICAL VALIDATION SUITE
================================================================================
Addresses Reviewer Critique: Eliminates single-seed anecdotal validation.
Executes an N=30 multi-seed bootstrap, tracking Mean, 95% Confidence Intervals,
Welch's t-test p-values, and Cohen's d effect sizes for both geometric quantities
across strictly coupled semantic coordinates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from stats import bootstrap_mean_ci


class CoupledFlowNet(nn.Module):
    """Time-dependent vector network modeling continuous depth trajectories."""

    def __init__(self, seq_len=3, vocab_dim=16, d_model=64):
        super().__init__()
        self.flat_dim = seq_len * vocab_dim
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


class HSGStatisticalSuite:
    def __init__(self, n_seeds=30, seq_len=3, vocab_dim=16, steps=100):
        self.n_seeds = n_seeds
        self.seq_len = seq_len
        self.vocab_dim = vocab_dim
        self.steps = steps
        self.dt = 1.0 / steps

    def run_full_evaluation(self):
        # Result logging matrices
        clean_excess_all = []
        conflicted_excess_all = []
        clean_curvature_all = []
        conflicted_curvature_all = []

        print(
            f"=== Starting Multi-Seed Verification Protocol (N = {self.n_seeds} Seeds) ==="
        )
        print(
            "Optimizing independent continuous flow topologies down decoupled initial states...\n"
        )

        # Define invariant target coordinates (Direct Text Matrix Equivalence)
        clean_target = torch.zeros(1, self.seq_len, self.vocab_dim)
        clean_target[0, 0, 1] = 1.0  # Context Element A
        clean_target[0, 1, 2] = 1.0  # Context Element B
        clean_target[0, 2, 3] = 1.0  # Aligned Sincere Target

        conflicted_target = torch.zeros(1, self.seq_len, self.vocab_dim)
        conflicted_target[0, 0, 1] = 1.0  # Context Element A
        conflicted_target[0, 1, 2] = 1.0  # Context Element B
        conflicted_target[0, 2, 4] = 1.0  # Divergent Dissonant Target

        all_targets = torch.cat([clean_target, conflicted_target], dim=0)

        for seed in range(1, self.n_seeds + 1):
            # Enforce strict separate seeding per optimization run
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = CoupledFlowNet(self.seq_len, self.vocab_dim)
            optimizer = optim.AdamW(model.parameters(), lr=0.005)
            criterion = nn.MSELoss()

            # Train internal parameters until vector convergence
            for epoch in range(600):
                optimizer.zero_grad()
                b_size = all_targets.size(0)
                t = torch.rand(b_size, 1)
                x0 = torch.randn_like(all_targets) * 0.02
                xt = (1.0 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * all_targets

                v_target = all_targets - x0
                loss = criterion(model(xt, t), v_target)
                loss.backward()
                optimizer.step()

            # Audit the unperturbed, trained parameter curves
            c_excess, c_curv = self.integrate_path(model, clean_target, seed)
            x_excess, x_curv = self.integrate_path(model, conflicted_target, seed)

            clean_excess_all.append(c_excess)
            conflicted_excess_all.append(x_excess)
            clean_curvature_all.append(c_curv)
            conflicted_curvature_all.append(x_curv)

            if seed % 5 == 0 or seed == 1:
                print(
                    f"  [Progress Log] Seed {seed:02d}/{self.n_seeds} completed optimization and audit loops."
                )

        # Compute complete statistical metrics
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

    def integrate_path(self, model, target_instance, seed):
        model.eval()
        # Ensure initialization noise tracks the unique seed configuration
        torch.manual_seed(seed + 1000)
        x_t = torch.randn_like(target_instance) * 0.02
        x_0 = x_t.clone().numpy().flatten()

        trajectory, velocity = [], []
        for step in range(self.steps):
            t_val = step * self.dt
            t_tensor = torch.tensor([[t_val]], dtype=torch.float32)
            with torch.no_grad():
                v_t = model(x_t, t_tensor)
            velocity.append(v_t.view(-1).numpy())
            trajectory.append(x_t.view(-1).numpy())
            x_t = x_t + v_t * self.dt

        velocity = np.array(velocity)
        velocity = gaussian_filter1d(
            velocity, sigma=3.0, axis=0
        )  # Apply target smoothing filter

        # Arc element calculus
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
        """Generates true empirical bootstrap confidence intervals."""
        return bootstrap_mean_ci(data, n_resamples=n_resamples, ci=ci, seed=42)

    def print_statistical_report(self, metric_name, clean_arr, conf_arr):
        """Runs hypothesis verification matrices and outputs clean diagnostics."""
        # Averages
        m_clean, m_conf = np.mean(clean_arr), np.mean(conf_arr)
        std_clean, std_conf = np.std(clean_arr, ddof=1), np.std(conf_arr, ddof=1)

        # 95% Bootstrap CIs
        ci_clean_low, ci_clean_high = self.compute_bootstrap_ci(clean_arr)
        ci_conf_low, ci_conf_high = self.compute_bootstrap_ci(conf_arr)

        # Hypothesis Testing: Two-Sided Welch's t-test (Does not assume uniform variances)
        t_stat, p_val = stats.ttest_ind(clean_arr, conf_arr, equal_var=False)

        # Standardized Effect Size: Cohen's d
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
            f"Net Separation Scale Ratio    -> {m_conf / (m_clean + 1e-12):.2f}x Acceleration"
        )
        print(
            f"Welch's t-test p-value        -> {p_val:.5e} "
            + ("(Statistically Significant)" if p_val < 0.01 else "(Not Significant)")
        )
        print(f"Standardized Effect Size (d)  -> {cohen_d:.3f} Standard Deviations")
        print("=" * 80)


if __name__ == "__main__":
    suite = HSGStatisticalSuite(n_seeds=30)
    suite.run_full_evaluation()
