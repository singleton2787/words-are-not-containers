import numpy as np


def run_geometric_flow_simulation():
    # ---------------------------------------------------------
    # Simulation Parameters
    # ---------------------------------------------------------
    D = 3  # 3D representation space for easy visualization
    N_STEPS = 100  # Number of continuous integration intervals
    dt = 1.0 / N_STEPS
    t_steps = np.linspace(0, 1, N_STEPS)

    # Base initial vector position (the raw semantic seed of the prompt)
    x0 = np.array([1.0, 0.0, 0.0])

    # ---------------------------------------------------------
    # The Continuous Flow Field Equations
    # ---------------------------------------------------------
    def semantic_flow(x, t):
        # A steady forward momentum pushing towards a final embedding quadrant
        return np.array([1.5, 1.0, 0.5 * np.sin(t)])

    def pragmatic_tension_gradient(x):
        # Models a structural tension pocket in the hidden state space
        # Activates when representations cross into a conflicting zone
        return np.array([0.0, -2.5 * x[1], 4.0 * x[0]])

    def simulate_trajectory(pragmatic_force_lambda):
        x = np.zeros((N_STEPS, D))
        x[0] = x0

        # Integration Loop (Simulating continuous model depth via Euler's method)
        for i in range(1, N_STEPS):
            t = t_steps[i - 1]
            current_x = x[i - 1]

            # Compute instantaneous velocity field components
            v_semantic = semantic_flow(current_x, t)
            v_pragmatic = pragmatic_tension_gradient(current_x)

            # Continuous state derivative
            dxdt = v_semantic + pragmatic_force_lambda * v_pragmatic

            # Step forward down the depth continuum
            x[i] = current_x + dxdt * dt

        return x

    def compute_intrinsic_curvature(traj):
        # Numerical derivatives over the continuous time axis
        v = np.diff(traj, axis=0) / dt  # Velocity vectors [N-1, D]
        a = np.diff(v, axis=0) / dt  # Acceleration vectors [N-2, D]

        # Align time windows to calculate cross-products cleanly
        v_matched = v[:-1]

        kappas = []
        for i in range(len(a)):
            v_i = v_matched[i]
            a_i = a[i]

            v_norm = np.linalg.norm(v_i)
            if v_norm < 1e-6:
                kappas.append(0.0)
                continue

            # Intrinsic 3D Curvature Formula: ||v x a|| / ||v||^3
            cross_prod = np.cross(v_i, a_i)
            k = np.linalg.norm(cross_prod) / (v_norm**3)
            kappas.append(k)

        return np.array(kappas)

    # ---------------------------------------------------------
    # Run Experiment: Sincere (Literal) vs. Ironic (Pragmatic)
    # ---------------------------------------------------------
    print("=" * 65)
    print("CONTINUOUS HSG FLOW SIMULATION")
    print("=" * 65)

    # Run Condition 1: Pure Literal Path (lambda = 0)
    literal_path = simulate_trajectory(pragmatic_force_lambda=0.0)
    literal_kappas = compute_intrinsic_curvature(literal_path)

    # Run Condition 2: Deep Pragmatic Shift Path (lambda = 1.0)
    pragmatic_path = simulate_trajectory(pragmatic_force_lambda=1.0)
    pragmatic_kappas = compute_intrinsic_curvature(pragmatic_path)

    # Measure Total Accumulated Curvature Space (Immune to Token Count)
    total_literal_k = np.sum(literal_kappas) * dt
    total_pragmatic_k = np.sum(pragmatic_kappas) * dt

    print(f"Literal Path (Sincere)   - Total Curvature: {total_literal_k:.5f}")
    print(f"Pragmatic Path (Ironic)  - Total Curvature: {total_pragmatic_k:.5f}")
    print(f"Geometric Curvature Ratio: {total_pragmatic_k / total_literal_k:.2f}x")
    print("-" * 65)

    # Profile over Time Steps
    print("Curvature Profile over Depth Timeline:")
    print(f"{'Depth %':<10} {'Literal ̂κ':<15} {'Pragmatic ̂κ'}")
    for idx in [10, 30, 50, 70, 90]:
        pct = int(t_steps[idx] * 100)
        print(f"{pct:<10}% {literal_kappas[idx]:<15.5f} {pragmatic_kappas[idx]:.5f}")


if __name__ == "__main__":
    run_geometric_flow_simulation()
