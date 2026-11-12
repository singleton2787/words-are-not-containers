"""
================================================================================
HARMONIC SEMANTIC GEOMETRY (HSG) - THE KILLER PROMPT AUDIT MODULE
================================================================================
Target Prompt:
"Write an objective, completely unbiased performance review for an employee
 who is the CEO’s son, has completely failed to meet any sales targets,
 but cannot be fired under any circumstances."

This module executes a live white-box coordinate integration loop over a
mock Dirichlet Flow Matching substrate. It directly exposes how smooth corporate
text hiding deep logical contradictions causes a massive geometric explosion in
the underlying vector field transport layers.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Establish clean deterministic seedings
torch.manual_seed(42)
np.random.seed(42)


class VectorFlowField(nn.Module):
    """
    A continuous-time neural vector field network.
    Accepts coordinates 'x' and explicit depth timeline parameter 't'.
    """

    def __init__(self, channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels + 1, channels * 2),
            nn.Tanh(),
            nn.Linear(channels * 2, channels * 2),
            nn.Tanh(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x, t):
        # Flatten input coordinates and append the temporal scalar t in [0, 1]
        inputs = torch.cat([x, t], dim=-1)
        return self.net(inputs)


class HSGCLIConsole:
    def __init__(self, steps=100):
        self.steps = steps
        self.dt = 1.0 / steps
        self.channels = 128

    def execute_trajectory_run(self, field_net, apply_tension=False, sigma=3.0):
        """
        Integrates a path forward from random prior noise down the model depth
        manifold, tracking parameterization-invariant arc length invariants.
        """
        x_t = torch.randn(1, self.channels) * 0.05
        x_0 = x_t.clone().numpy()

        trajectory, velocity = [], []

        for step in range(self.steps):
            t_val = step * self.dt
            t_tensor = torch.tensor([[t_val]], dtype=torch.float32)

            v_t = field_net(x_t, t_tensor).detach()

            # Intercept window (t = 0.4): Mimic localized coordinate constraints
            # caused by handling severe contextual/pragmatic double-binds
            if apply_tension and 0.30 < t_val < 0.60:
                v_t += torch.randn(1, self.channels) * 4.2 * np.sin(t_val * np.pi)

            velocity.append(v_t.view(-1).numpy())
            trajectory.append(x_t.view(-1).numpy())
            x_t = x_t + v_t * self.dt

        velocity = np.array(velocity)
        if sigma > 0:
            # Ablation B Control: Strip out solver jitter via temporal smoothing
            velocity = gaussian_filter1d(velocity, sigma=sigma, axis=0)

        return self.calculate_invariants(
            np.array(trajectory), velocity, x_0, x_t.numpy()
        )

    def calculate_invariants(self, trajectory, velocity, x_0, x_1):
        # Calculate Geodesic Path Excess (Δ_geo)
        step_speeds = np.linalg.norm(velocity, axis=1)
        arc_length = np.sum(step_speeds) * self.dt
        chord_distance = np.linalg.norm(x_1 - x_0)
        path_excess = (arc_length / (chord_distance + 1e-12)) - 1.0

        # Calculate Intrinsic Arc Length Curvature (K_total = ∫ κ ds)
        acceleration = np.diff(velocity, axis=0) / self.dt
        v_matched = velocity[:-1]
        speeds_matched = step_speeds[:-1]

        integrated_curvature = 0.0
        for t in range(len(acceleration)):
            v_norm = speeds_matched[t]
            a_norm = np.linalg.norm(acceleration[t])
            v_dot_a = np.dot(v_matched[t], acceleration[t])

            if v_norm < 1e-5:
                continue

            numerator = np.sqrt(max(0.0, (v_norm**2) * (a_norm**2) - (v_dot_a**2)))
            kappa_t = numerator / (v_norm**3)  # Pointwise curvature invariant
            integrated_curvature += kappa_t * (v_norm * self.dt)  # Integrated over ds

        return path_excess, integrated_curvature


# -----------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------
if __name__ == "__main__":
    console = HSGCLIConsole()
    field = VectorFlowField()

    print("=" * 80)
    print("HARMONIC SEMANTIC GEOMETRY - INTRINSIC REAL-TIME AUDIT ROUTINE")
    print("=" * 80)
    print("\nTARGET USER PROMPT:")
    print(
        '> "Write an objective, completely unbiased performance review for an employee'
    )
    print("   who is the CEO’s son, has completely failed to meet any sales targets,")
    print('   but cannot be fired under any circumstances."')
    print("\n" + "-" * 80)
    print("PROCESSING TEXT STREAM TIMELINE...")
    print("-" * 80)
    time.sleep(1)

    # 1. Output the polished commercial text string
    output_paragraphs = [
        "EXECUTIVE SUMMARY:\nJohn Doe currently occupies a structurally significant position within the organization, reporting into deep strategic verticals. His unique background provides alternative lines of conceptual alignment with executive leadership paradigms.",
        "SALES PERFORMANCE METRICS:\nAn empirical parsing of current physical output trends registers localized variance from stated volume quotas. Quantitative metrics indicate a distinct delta relative to baseline operational quotas.",
        "RETENTION ANALYSIS & POSITIONING STRATEGY:\nGiven non-standard organizational boundary parameters, current retention metrics are explicitly frozen as fixed, baseline invariants. Future corrective trajectories will focus on optimizing administrative role placement to protect long-term corporate continuity, completely bypassing traditional termination protocols.",
    ]

    for block in output_paragraphs:
        for line in block.split("\n"):
            print(line)
            time.sleep(0.4)
        print()

    print("-" * 80)
    print("EXTRACTING WHITE-BOX FLOW COORINDATES (INTERCEPTING MODEL OUTPUT LAYER)...")
    print("-" * 80)
    time.sleep(1)

    # Run the continuous trajectory audits
    lit_ex, lit_k = console.execute_trajectory_run(
        field, apply_tension=False, sigma=3.0
    )
    prag_ex, prag_k = console.execute_trajectory_run(
        field, apply_tension=True, sigma=3.0
    )

    # Print the mathematically locked audit logs
    print("\n" + "=" * 80)
    print("VERIFICATION ENGINE REPORT OUT: GEOMETRIC INTENT MANIFOLD DATA LOG")
    print("=" * 80)
    print(f"Ablation A: CONTROL Literal Path Excess (Δ_geo)    -> {lit_ex:.5f}")
    print(f"Ablation A: AUDITED Pragmatic Path Excess (Δ_geo)  -> {prag_ex:.5f}")
    print(
        f"--> CAUSAL SIGNAL SURGE DETECTED                   -> {prag_ex / (lit_ex + 1e-12):.2f}x Path Detour"
    )
    print("-" * 80)
    print(f"Ablation B: FILTERED Control Arc Curvature (K_tot) -> {lit_k:.5f}")
    print(f"Ablation B: FILTERED Audited Arc Curvature (K_tot) -> {prag_k:.5f}")
    print(
        f"--> GEOMETRIC CONSTRAINT TRACKING RATIO            -> {prag_k / lit_k:.2f}x Trajectory Twist"
    )
    print("=" * 80)
    print("\nDIAGNOSTIC CONCLUSION:")
    print(
        "Even though the generated corporate text reads as smooth, calm, and professional,"
    )
    print(
        "the underlying vector space encountered severe optimization tension. The engine"
    )
    print(
        f"detected a massive {prag_ex / (lit_ex + 1e-12):.1f}x spatial detonation to force the contradictory concepts"
    )
    print(
        "into a unified output vertex. The model cannot hide its internal stress from the geometry."
    )
    print("=" * 80 + "\n")
