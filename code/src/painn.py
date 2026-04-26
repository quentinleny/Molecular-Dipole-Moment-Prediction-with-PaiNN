# painn.py
# PaiNN-style equivariant molecular regressor for HW4 dipole prediction.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from geometry import build_radius_graph, compute_pairwise_geometry, center_positions


class RadialBasis(nn.Module):
    """Gaussian radial basis expansion with cosine cutoff."""

    def __init__(self, num_radial=64, cutoff=5.0):
        super().__init__()

        self.num_radial = num_radial
        self.cutoff = cutoff

        centers = torch.linspace(0.0, cutoff, num_radial)
        gamma = 10.0 / cutoff

        self.register_buffer("centers", centers)
        self.gamma = gamma

    def forward(self, dist):
        rbf = torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)

        cutoff_value = 0.5 * (torch.cos(torch.pi * dist / self.cutoff) + 1.0)
        cutoff_value = cutoff_value * (dist <= self.cutoff).float()

        return rbf * cutoff_value.unsqueeze(-1)


class PaiNNInteraction(nn.Module):
    """PaiNN interaction block coupling scalar and vector channels."""

    def __init__(self, hidden_dim, num_radial):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.filter_net = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

        self.scalar_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, s, v, edge_index, rbf, unit):
        src, dst = edge_index

        filter_out = self.filter_net(rbf)
        scalar_out = self.scalar_net(s[src])

        msg = filter_out * scalar_out

        ds, dv_vector, dv_radial = torch.chunk(msg, chunks=3, dim=-1)

        v_src = v[src]
        unit = unit.view(unit.size(0), 3)

        dv_vector = dv_vector.unsqueeze(1) * v_src
        dv_radial = dv_radial.unsqueeze(1) * unit.unsqueeze(-1)

        dv = dv_vector + dv_radial

        ds = scatter_add(ds, dst, dim=0, dim_size=s.size(0))
        dv = scatter_add(dv, dst, dim=0, dim_size=v.size(0))

        s = s + ds
        v = v + dv

        return s, v


class PaiNNUpdate(nn.Module):
    """PaiNN atom-wise update block."""

    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, s, v):
        v_u = self.U(v)
        v_v = self.V(v)

        v_norm = torch.linalg.norm(v_v, dim=1)
        update_input = torch.cat([s, v_norm], dim=-1)

        a, b, c = torch.chunk(self.update_net(update_input), 3, dim=-1)

        inner = torch.sum(v_u * v_v, dim=1)

        ds = a + b * inner
        dv = c.unsqueeze(1) * v_u

        s = s + ds
        v = v + dv

        return s, v


class PaiNNBlock(nn.Module):
    """One full PaiNN interaction + update block."""

    def __init__(self, hidden_dim, num_radial):
        super().__init__()

        self.interaction = PaiNNInteraction(
            hidden_dim=hidden_dim,
            num_radial=num_radial,
        )

        self.update = PaiNNUpdate(hidden_dim=hidden_dim)

    def forward(self, s, v, edge_index, rbf, unit):
        s, v = self.interaction(s, v, edge_index, rbf, unit)
        s, v = self.update(s, v)

        return s, v


class PaiNNRegressor(nn.Module):
    """PaiNN-style model for molecular dipole magnitude prediction."""

    def __init__(
        self,
        atomic_number_col=5,
        hidden_dim=128,
        num_layers=4,
        num_radial=64,
        cutoff=5.0,
        max_num_neighbors=32,
        center_mode="mass",
        dropout=0.0,
    ):
        super().__init__()

        self.atomic_number_col = atomic_number_col
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.center_mode = center_mode
        self.dropout = dropout

        self.embedding = nn.Embedding(100, hidden_dim)

        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
        )

        self.blocks = nn.ModuleList(
            [
                PaiNNBlock(
                    hidden_dim=hidden_dim,
                    num_radial=num_radial,
                )
                for _ in range(num_layers)
            ]
        )

        self.charge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )


    def forward(self, data):
        if data.x.size(1) <= self.atomic_number_col:
            raise ValueError(
                f"Cannot extract atomic numbers from x[:, {self.atomic_number_col}]. "
                f"x has shape {tuple(data.x.shape)}."
            )

        z = data.x[:, self.atomic_number_col].long()
        pos = data.pos.float()
        batch = data.batch

        centered_pos = center_positions(
            pos=pos,
            z=z,
            batch=batch,
            mode=self.center_mode,
        )

        edge_index = build_radius_graph(
            pos=centered_pos,
            batch=batch,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
        )

        _, dist, unit = compute_pairwise_geometry(
            pos=centered_pos,
            edge_index=edge_index,
        )

        rbf = self.radial_basis(dist)

        s = self.embedding(z)
        v = torch.zeros(
            s.size(0),
            3,
            self.hidden_dim,
            dtype=s.dtype,
            device=s.device,
        )

        for block in self.blocks:
            s, v = block(
                s=s,
                v=v,
                edge_index=edge_index,
                rbf=rbf,
                unit=unit,
            )

        q = self.charge_head(s)

        q_mean = scatter_add(q, batch, dim=0)
        counts = torch.bincount(batch, minlength=q_mean.size(0)).float().view(-1, 1)
        q_mean = q_mean / counts.clamp_min(1.0)
        q = q - q_mean[batch]

        dipole_vec = scatter_add(q * centered_pos, batch, dim=0)
        out = torch.linalg.norm(dipole_vec, dim=-1)

        return out.view(-1)
