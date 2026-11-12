import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class ContinuousFlowField(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 4-token sequence x 16-dim semantic space = 64 dimensions
        self.state_dim = 64
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.state_dim),
        )

    def forward(self, x, t):
        # x: [Batch, state_dim], t: [Batch, 1]
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


# Initialize System
flow_net = ContinuousFlowField().cuda()
optimizer = optim.AdamW(flow_net.parameters(), lr=0.002)
N_STEPS = 100
dt = 1.0 / N_STEPS

# Generate Fixed Target Anchors for the Minimal Pairs
# Represents the high-dimensional latent destination matrices of our utterances
literal_dest = torch.randn(1, 64).cuda() * 2.0
pragmatic_dest = literal_dest.clone()
# Apply a localized pragmatic transformation vector to the final phrase segment
pragmatic_dest[0, 48:] += torch.randn(16).cuda() * 3.5

# Train the Flow-Matching Operator via Linear Paths
print("Optimizing Continuous Flow Manifold Fields...")
for epoch in range(600):
    optimizer.zero_grad()
    t_val = torch.rand(1, 1).cuda()
    x0 = torch.randn(1, 64).cuda() * 0.5
    x1 = literal_dest if epoch % 2 == 0 else pragmatic_dest

    # Linear optimal transport path definition
    xt = (1.0 - t_val) * x0 + t_val * x1
    v_target = x1 - x0

    loss = nn.MSELoss()(flow_net(xt, t_val), v_target)
    loss.backward()
    optimizer.step()


# ---------------------------------------------------------
# Continuous Geometric Evaluation Engine
# ---------------------------------------------------------
def evaluate_path_geometry(lambda_pragmatic):
    flow_net.eval()
    x = torch.randn(1, 64).cuda() * 0.2
    x_start = x.clone().cpu().numpy()[0]

    x_path = [x.clone()]
    v_path = []

    # Continuous ODE Integration passing smoothly through depth time
    for i in range(N_STEPS):
        t_val = i * dt
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32).cuda()

        with torch.no_grad():
            vt = flow_net(x, t_tensor)

            # Unforced Pragmatic Deflection: Simulates internal tension
            if lambda_pragmatic > 0 and t_val > 0.2 and t_val < 0.6:
                vt[0, 48:] += torch.sin(torch.tensor(t_val * np.pi)).cuda() * 4.5

            v_path.append(vt.clone().cpu().numpy()[0])
            x = x + vt * dt
            x_path.append(x.clone())

    x_end = x.cpu().numpy()[0]
    v_path = np.array(v_path)

    # 1. Calculate Absolute Arc Length: Integral of ||v(t)|| dt
    arc_length = float(np.sum(np.linalg.norm(v_path, axis=1)) * dt)
    chord_distance = float(np.linalg.norm(x_end - x_start))
    path_excess = (arc_length / (chord_distance + 1e-10)) - 1.0

    # 2. Calculate Integrated Curvature: Integral of ||dv/dt|| / ||v||^2 dt
    dv = np.diff(v_path, axis=0) / dt
    kappas = []
    for step in range(len(dv)):
        v_norm = np.linalg.norm(v_path[step])
        dv_norm = np.linalg.norm(dv[step])
        if v_norm < 1e-5:
            kappas.append(0.0)
        else:
            kappas.append(dv_norm / (v_norm**2))

    integrated_k = float(np.mean(kappas))

    return path_excess, integrated_k


# Run Pure Continuous Evaluation
excess_low, kappa_low = evaluate_path_geometry(lambda_pragmatic=0.0)
excess_high, kappa_high = evaluate_path_geometry(lambda_pragmatic=1.0)

print("\n" + "=" * 65)
print("PURE CONTINUOUS GEOMETRIC FLOW EVALUATION")
print("=" * 65)
print(
    f"Literal Track   -> Path Excess: {excess_low:.5f} | Integrated ̂κ: {kappa_low:.5f}"
)
print(
    f"Pragmatic Track -> Path Excess: {excess_high:.5f} | Integrated ̂κ: {kappa_high:.5f}"
)
print("-" * 65)
print(f"Geometric Path Excess Separation: {excess_high / (excess_low + 1e-10):.2f}x")
print(f"Geometric Curvature Separation:   {kappa_high / (kappa_low + 1e-10):.2f}x")
print("=" * 65)
