import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d


class HSGProductionPipeline:
    """
    Continuous-Time Trajectory Auditor for Simplex-Based Language Models.
    Designed for deployment within production Flow Matching and Neural ODE substrates.
    """

    def __init__(self, integration_steps=100, sigma_ablation=3.0):
        """
        Args:
            integration_steps: Number of forward intervals used by the ODE solver.
            sigma_ablation: Smoothness kernel variance for local noise isolation.
        """
        self.steps = integration_steps
        self.dt = 1.0 / integration_steps
        self.sigma = sigma_ablation

    @torch.no_grad()
    def audit_inference_trajectory(
        self, model, conditioning_context, vocabulary_size, sequence_length
    ):
        """
        Hooks directly into a production model's forward sampling pass.

        Args:
            model: Active flow-matching checkpoint exposing `compute_velocity_field(x, t, context)`
            conditioning_context: Latent embeddings of the prompt prefix
            vocabulary_size (V): Dimensionality of the token simplex
            sequence_length (L): Total length of the allocation window
        """
        model.eval()
        device = next(model.parameters()).device

        # Initialize trajectory at the center of the simplex product (maximum entropy noise)
        x_t = torch.full(
            (1, sequence_length, vocabulary_size), 1.0 / vocabulary_size, device=device
        )

        trajectory_log = []
        velocity_log = []

        # -----------------------------------------------------------------
        # Core ODE Intercept Window
        # -----------------------------------------------------------------
        for step in range(self.steps):
            t_val = step * self.dt
            t_tensor = torch.tensor([[t_val]], dtype=torch.float32, device=device)

            # Extract instantaneous velocity field vector pointing along the simplex tangent plane
            v_t = model.compute_velocity_field(x_t, t_tensor, conditioning_context)

            # Record flattened states to preserve full multi-token cross-correlations
            velocity_log.append(v_t.view(-1).cpu().numpy())
            trajectory_log.append(x_t.view(-1).cpu().numpy())

            # Execute standard continuous integration step down the depth continuum
            x_t = x_t + v_t * self.dt

        trajectory = np.array(trajectory_log)
        velocity = np.array(velocity_log)

        # -----------------------------------------------------------------
        # Operational Execution of Ablation B (Temporal Noise Filtering)
        # -----------------------------------------------------------------
        if self.sigma > 0:
            velocity = gaussian_filter1d(velocity, sigma=self.sigma, axis=0)

        return self._compute_geometric_invariants(trajectory, velocity)

    def _compute_geometric_invariants(self, trajectory, velocity):
        """
        Executes the formal differential geometry tracking calculations.
        """
        # Metric 1: Geodesic Path Excess (Δ_geo)
        step_magnitudes = np.linalg.norm(velocity, axis=1)
        integrated_arc_length = np.sum(step_magnitudes) * self.dt

        x_0 = trajectory[0]
        x_1 = trajectory[-1]
        chord_distance = np.linalg.norm(x_1 - x_0)

        path_excess = (integrated_arc_length / (chord_distance + 1e-12)) - 1.0

        # Metric 2: Total Integrated Curvature (K_total)
        acceleration = np.diff(velocity, axis=0) / self.dt
        v_matched = velocity[:-1]

        kappas = []
        for t in range(len(acceleration)):
            v_norm = np.linalg.norm(v_matched[t])
            a_norm = np.linalg.norm(acceleration[t])

            if v_norm < 1e-6:
                kappas.append(0.0)
            else:
                # Intrinsic curvature scaled inversely by kinetic energy
                kappas.append(a_norm / (v_norm**2))

        return {
            "geodesic_path_excess": float(path_excess),
            "total_integrated_curvature": float(np.mean(kappas)),
        }
