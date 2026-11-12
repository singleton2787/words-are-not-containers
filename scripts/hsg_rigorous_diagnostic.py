import numpy as np


def compute_rigorous_hsg_metrics(trajectory, velocity, dt=0.01):
    """
    Inputs:
        trajectory: NumPy array of shape [N_STEPS, TOTAL_DIMENSIONS]
        velocity:   NumPy array of shape [N_STEPS, TOTAL_DIMENSIONS]
    """
    # ---------------------------------------------------------
    # Metric 1: Absolute Geodesic Path Excess (Δ_geo)
    # ---------------------------------------------------------
    # Integrate vector field magnitudes: ∫ ||v(t)|| dt
    step_lengths = np.linalg.norm(velocity, axis=1)
    integrated_arc_length = np.sum(step_lengths) * dt

    # Calculate straight-line Euclidean distance between boundaries
    x_0 = trajectory[0]
    x_1 = trajectory[-1]
    chord_distance = np.linalg.norm(x_1 - x_0)

    # Invariant Path Excess Definition
    path_excess = (integrated_arc_length / (chord_distance + 1e-12)) - 1.0

    # ---------------------------------------------------------
    # Metric 2: Integrated Trajectory Curvature (K_total)
    # ---------------------------------------------------------
    # Numerical derivative of velocity with respect to time axis (dv/dt)
    acceleration = np.diff(velocity, axis=0) / dt
    v_matched = velocity[:-1]

    kappas = []
    for t in range(len(acceleration)):
        v_norm = np.linalg.norm(v_matched[t])
        a_norm = np.linalg.norm(acceleration[t])

        if v_norm < 1e-6:
            kappas.append(0.0)
        else:
            # Native intrinsic curvature: acceleration magnitude scaled by kinetic energy
            kappas.append(a_norm / (v_norm**2))

    total_curvature = np.mean(kappas)

    return {
        "path_excess": float(path_excess),
        "integrated_curvature": float(total_curvature),
    }


# Scientific Verification Matrix Example
if __name__ == "__main__":
    print("HSG Rigorous Diagnostic Engine Initialized.")
    print("Ready to process intercepted vector streams from Dirichlet Flow targets.")
