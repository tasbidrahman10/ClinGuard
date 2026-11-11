"""Outer (coordinator) step. Conventions are fixed by decision D7 / manual Eq. (1), (2), (6):

    d_i   = phi_t - phi_i^(H)                      pseudo-gradient of site i
    d_bar = sum_i w_i d_i
    m_t   = mu * m_{t-1} + d_bar                   m_{-1} = 0
    phi_{t+1} = phi_t - eta_out * (d_bar + mu * m_t)

FedAvg is the same step with eta_out = 1, mu = 0 (phi_{t+1} = sum_i w_i phi_i).
All functions work on flat 1-D tensors and never modify their inputs.
"""
import torch


def pseudo_gradient(phi_start, phi_local):
    return phi_start - phi_local


def aggregate(deltas, weights):
    """Weighted sum of the sites' pseudo-gradients. weights must be on the simplex."""
    if len(deltas) != len(weights):
        raise ValueError(f"{len(deltas)} deltas but {len(weights)} weights")
    w = torch.as_tensor(weights, dtype=deltas[0].dtype)
    if (w < 0).any() or abs(w.sum().item() - 1.0) > 1e-6:
        raise ValueError(f"weights must be non-negative and sum to 1, got {w.tolist()}")
    d_bar = torch.zeros_like(deltas[0])
    for wi, d in zip(w, deltas):
        d_bar = d_bar + wi * d
    return d_bar


def nesterov_step(phi, m, d_bar, eta_out, mu):
    """Returns (phi_next, m_next)."""
    m_next = mu * m + d_bar
    phi_next = phi - eta_out * (d_bar + mu * m_next)
    return phi_next, m_next


def nesterov_direction(m_prev, d_bar, mu):
    """v_t in manual Eq. (6): the step actually taken is phi_{t+1} = phi_t - eta_out * v_t."""
    return mu ** 2 * m_prev + (1 + mu) * d_bar
