# The Core Intercept Loop for Production Flow Matching Engines
import torch
import numpy as np


def hsg_continuous_interceptor(model, initial_noise, target_tokens, steps=100):
    dt = 1.0 / steps
    x_t = initial_noise.clone()

    trajectory_states = []
    velocity_vectors = []

    for step in range(steps):
        t_val = step * dt
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32).cuda()

        # Intercept the production model's instantaneous velocity field
        with torch.no_grad():
            # In Dirichlet FM, the vector field points along the simplex tangent plane
            v_t = model.compute_velocity_field(x_t, t_tensor)

        velocity_vectors.append(v_t.cpu().numpy())
        trajectory_states.append(x_t.cpu().numpy())

        # Core integration step forward down the depth continuum
        x_t = x_t + v_t * dt

    return np.array(trajectory_states), np.array(velocity_vectors)
