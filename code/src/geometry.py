# geometry.py
# Geometry utilities for equivariant molecular models (PaiNN).

import torch
from torch_geometric.nn import radius_graph
import math
from torch_scatter import scatter_mean, scatter_sum


# Atomic masses for QM9 elements: H, C, N, O, F
# Using standard atomic weights (approximate, sufficient for center-of-mass)
_ATOMIC_MASS = {
    1: 1.0079,   # H
    6: 12.011,   # C
    7: 14.0067,  # N
    8: 15.999,   # O
    9: 18.998,   # F
}


def atomic_masses_from_z(z: torch.Tensor) -> torch.Tensor:
    """
    Map atomic numbers to atomic masses.

    Args:
        z: LongTensor [N]

    Returns:
        masses: FloatTensor [N]
    """
    device = z.device
    masses = torch.zeros_like(z, dtype=torch.float, device=device)

    for atomic_number, mass in _ATOMIC_MASS.items():
        masses[z == atomic_number] = mass

    return masses

def center_positions(
    pos: torch.Tensor,
    batch: torch.Tensor,
    z: torch.Tensor = None,
    mode: str = "mass",
) -> torch.Tensor:
    """
    Center positions per graph.

    Args:
        pos:   [N, 3]
        batch: [N]
        z:     [N] atomic numbers (required for mass mode)
        mode:  "mass" or "mean"

    Returns:
        centered_pos: [N, 3]
    """

    if mode not in ["mass", "mean"]:
        raise ValueError(f"Unsupported centering mode: {mode}")

    if mode == "mean":
        center = scatter_mean(pos, batch, dim=0)
        return pos - center[batch]

    # mass-based centering
    if z is None:
        raise ValueError("z required for mass-based centering")

    masses = atomic_masses_from_z(z)  # [N]
    mass_sum = scatter_sum(masses, batch, dim=0)  # [B]
    weighted_pos = pos * masses.unsqueeze(-1)     # [N, 3]
    center = scatter_sum(weighted_pos, batch, dim=0) / mass_sum.unsqueeze(-1)

    return pos - center[batch]

def build_radius_graph(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    max_num_neighbors: int,
):
    """
    Build radius graph from positions.

    Args:
        pos: [N, 3]
        batch: [N]
        cutoff: float
        max_num_neighbors: int

    Returns:
        edge_index: [2, E]
    """
    edge_index = radius_graph(
        x=pos,
        r=cutoff,
        batch=batch,
        loop=False,
        max_num_neighbors=max_num_neighbors,
    )

    return edge_index

def compute_pairwise_geometry(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Compute displacement vectors and distances for edges.

    Args:
        pos: [N, 3]
        edge_index: [2, E]

    Returns:
        r_ij:  [E, 3]   displacement (source -> target)
        dist:  [E]
        unit:  [E, 3]   normalized direction vectors
    """
    src, dst = edge_index

    r_ij = pos[src] - pos[dst]        # [E, 3]
    dist = torch.linalg.norm(r_ij, dim=-1)  # [E]

    unit = r_ij / (dist.unsqueeze(-1) + eps)

    return r_ij, dist, unit

def random_rotation_matrix(device):
    """
    Generate a random 3x3 rotation matrix.
    Uses axis-angle sampling.
    """
    # Random unit axis
    axis = torch.randn(3, device=device)
    axis = axis / torch.linalg.norm(axis)

    # Random angle in [0, 2π]
    angle = 2 * math.pi * torch.rand(1, device=device)

    K = torch.tensor(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ],
        device=device,
    )

    I = torch.eye(3, device=device)

    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

    return R

def rotate_positions(pos: torch.Tensor, R: torch.Tensor):
    """
    Rotate positions by rotation matrix R.

    Args:
        pos: [N, 3]
        R:   [3, 3]

    Returns:
        rotated_pos: [N, 3]
    """
    return pos @ R.T

def neighbor_statistics(edge_index, batch):
    """
    Compute average neighbors per graph.

    Returns:
        avg_neighbors_per_graph: float
    """
    _, dst = edge_index
    counts = torch.bincount(batch[dst])
    return counts.float().mean().item()
