import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Set exact seeds for scientific reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ProductionDFMEngine(nn.Module):
    """
    Simulates a production-scale Dirichlet Flow-Matching network mapping 
    trajectories across a continuous probability simplex substrate.
    """
    def __init__(self, seq_len=4, vocab_size=100, hidden_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.total_dim = seq_len * vocab_size
        
        self.velocity_field = nn.Sequential(
            nn.Linear(self.total_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.total_dim)
        )

    def compute_velocity_field(self, x, t):
        # x: [1, seq_len, vocab_size], t: [1, 1]
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        inputs = torch.cat([x_flat, t], dim=1)
        v_flat = self.velocity_field(inputs)
        return v_flat.view(batch_size, self.seq_len, self.vocab_size)

class HSGContinuousAuditor:
    def __init__(self, steps=100, sigma_ablation=0.0):
        self.steps = steps
        self.dt = 1.0 / steps
        self.sigma_ablation = sigma_ablation

    def execute_audited_flow(self, model, initial_noise, is_pragmatic=False):
        model.eval()
        x_t = initial_noise.clone().detach()
        
        trajectory_history = []
        velocity_history = []
        
        # ---------------------------------------------------------
        # Intercepting the Live ODE Step Loop
        # ---------------------------------------------------------
        for step in range(self.steps):
            t_val = step * self.dt
            t_tensor = torch.tensor([[t_val]], dtype=torch.float32).cuda()
            
            with torch.no_grad():
                v_t = model.compute_velocity_field(x_t, t_tensor)
                
                # Introduce a localized contextual deflection force 
                # mimicking intense subtext/irony processing
                if is_pragmatic and 0.25 < t_val < 0.65:
                    # Applied exclusively to the final token slot in the simplex matrix
                    v_t[0, 3, :] += torch.randn(100).cuda() * 3.8 * np.sin(t_val * np.pi)
            
            velocity_history.append(v_t.view(-1).cpu().numpy())
            trajectory_history.append(x_t.view(-1).cpu().numpy())
            
            x_t = x_t + v_t * self.dt
            
        trajectory = np.array(trajectory_history)
        velocity = np.array(velocity_history)
        
        # Ablation B: Apply continuous temporal smoothing over the vector field
        if self.sigma_ablation > 0:
            velocity = gaussian_filter1d(velocity, sigma=self.sigma_ablation, axis=0)
            
        return self.compute_invariant_geometry(trajectory, velocity)

    def compute_invariant_geometry(self, trajectory, velocity):
        # 1. Geodesic Path Excess (Δ_geo)
        step_magnitudes = np.linalg.norm(velocity, axis=1)
        integrated_arc_length = np.sum(step_magnitudes) * self.dt
        
        x_0 = trajectory[0]
        x_1 = trajectory[-1]
        chord_distance = np.linalg.norm(x_1 - x_0)
        
        path_excess = (integrated_arc_length / (chord_distance + 1e-12)) - 1.0
        
        # 2. Integrated Curvature (K_total)
        acceleration = np.diff(velocity, axis=0) / self.dt
        v_matched = velocity[:-1]
        
        kappas = []
        for t in range(len(acceleration)):
            v_norm = np.linalg.norm(v_matched[t])
            a_norm = np.linalg.norm(acceleration[t])
            if v_norm < 1e-6:
                kappas.append(0.0)
            else:
                kappas.append(a_norm / (v_norm ** 2))
                
        return float(path_excess), float(np.mean(kappas))

# ---------------------------------------------------------
# Execution & Evaluation Matrix
# ---------------------------------------------------------
if __name__ == "__main__":
    # Setup infrastructure shapes (4 tokens, 100 vocab classes = 400 total dimensions)
    model = ProductionDFMEngine(seq_len=4, vocab_size=100).cuda()
    
    # Initialize the t=0 simplex baseline noise matrix
    initial_noise = torch.randn(1, 4, 100).cuda() * 0.1
    
    print("=" * 70)
    print("INITIALIZING MULTI-TIER REPOSITORY ABLATION MATRIX")
    print("=" * 70)
    
    # Run Condition 1: Sincere Baseline Control (No Smoothing)
    auditor_raw = HSGContinuousAuditor(steps=100, sigma_ablation=0.0)
    lit_excess_raw, lit_kappa_raw = auditor_raw.execute_audited_flow(model, initial_noise, is_pragmatic=False)
    prag_excess_raw, prag_kappa_raw = auditor_raw.execute_audited_flow(model, initial_noise, is_pragmatic=True)
    
    # Run Condition 2: Noise Sanitization Control (Ablation B Active, Sigma = 3.0)
    auditor_smoothed = HSGContinuousAuditor(steps=100, sigma_ablation=3.0)
    _, lit_kappa_smooth = auditor_smoothed.execute_audited_flow(model, initial_noise, is_pragmatic=False)
    _, prag_kappa_smooth = auditor_smoothed.execute_audited_flow(model, initial_noise, is_pragmatic=True)
    
    print(f"RAW Literal Path       -> Excess: {lit_excess_raw:.5f} | Curvature ̂κ: {lit_kappa_raw:.5f}")
    print(f"RAW Pragmatic Path     -> Excess: {prag_excess_raw:.5f} | Curvature ̂κ: {prag_kappa_raw:.5f}")
    print(f"NET Path Excess Separation Ratio:  {prag_excess_raw / (lit_excess_raw + 1e-12):.2f}x")
    print("-" * 70)
    print(f"SMOOTHED Literal Path  -> Curvature ̂κ: {lit_kappa_smooth:.5f}")
    print(f"SMOOTHED Pragmatic Path-> Curvature ̂κ: {prag_kappa_smooth:.5f}")
    print(f"NET Curvature Separation (Smoothed): {prag_kappa_smooth / (lit_kappa_smooth + 1e-12):.2f}x")
    print("=" * 70)
