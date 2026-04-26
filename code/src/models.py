# models.py
# GNN model definitions for HW4 molecular graph regression.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    GINEConv,
    GraphNorm,
    NNConv,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.models import SchNet
from painn import PaiNNRegressor


class GINEBlock(nn.Module):
    """Residual GINE block with normalization, ReLU, and dropout."""

    def __init__(self, hidden_dim, edge_dim, dropout=0.1, norm_type="batch"):
        super().__init__()

        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv = GINEConv(nn=mlp, edge_dim=edge_dim)

        if norm_type == "batch":
            self.norm = BatchNorm(hidden_dim)
        elif norm_type == "graph":
            self.norm = GraphNorm(hidden_dim)
        elif norm_type is None or norm_type == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch=None):
        residual = x

        x = self.conv(x, edge_index, edge_attr)

        if isinstance(self.norm, GraphNorm):
            x = self.norm(x, batch)
        else:
            x = self.norm(x)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x + residual

        return x

class GINERegressor(nn.Module):
    """GINE graph-level regressor for molecular property prediction."""

    def __init__(
        self,
        node_dim=11,
        edge_dim=5,
        hidden_dim=128,
        num_layers=4,
        dropout=0.1,
        pooling="mean_add_max",
        norm_type="batch",
        use_graph_features=False,
        graph_feat_dim=0,
    ):
        super().__init__()

        self.pooling = pooling
        self.use_graph_features = use_graph_features
        self.graph_feat_dim = graph_feat_dim

        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                GINEBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_layers)
            ]
        )

        if pooling == "set2set":
            self.set2set = Set2Set(hidden_dim, processing_steps=3)
        else:
            self.set2set = None

        pool_dim = self._get_pool_dim(hidden_dim)

        if use_graph_features:
            pool_dim += graph_feat_dim

        self.head = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _get_pool_dim(self, hidden_dim):
        if self.pooling == "mean":
            return hidden_dim
        if self.pooling == "add":
            return hidden_dim
        if self.pooling == "max":
            return hidden_dim
        if self.pooling == "mean_add":
            return 2 * hidden_dim
        if self.pooling == "mean_add_max":
            return 3 * hidden_dim
        if self.pooling == "set2set":
            return 2 * hidden_dim

        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def pool(self, x, batch):
        if self.pooling == "mean":
            return global_mean_pool(x, batch)

        if self.pooling == "add":
            return global_add_pool(x, batch)

        if self.pooling == "max":
            return global_max_pool(x, batch)

        if self.pooling == "mean_add":
            return torch.cat(
                [
                    global_mean_pool(x, batch),
                    global_add_pool(x, batch),
                ],
                dim=1,
            )

        if self.pooling == "mean_add_max":
            return torch.cat(
                [
                    global_mean_pool(x, batch),
                    global_add_pool(x, batch),
                    global_max_pool(x, batch),
                ],
                dim=1,
            )

        if self.pooling == "set2set":
            return self.set2set(x, batch)

        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = self.node_encoder(x)
        x = F.relu(x)

        for block in self.blocks:
            x = block(x, edge_index, edge_attr, batch=batch)

        x = self.pool(x, batch)

        if self.use_graph_features:
            if not hasattr(data, "graph_features"):
                raise ValueError("use_graph_features=True but data.graph_features is missing.")
            x = torch.cat([x, data.graph_features], dim=1)

        out = self.head(x)

        return out.view(-1)

class NNConvBlock(nn.Module):
    """Residual NNConv block with edge-conditioned messages."""

    def __init__(self, hidden_dim, edge_dim, dropout=0.1, norm_type="batch"):
        super().__init__()

        edge_network = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )

        self.conv = NNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            nn=edge_network,
            aggr="mean",
        )

        if norm_type == "batch":
            self.norm = BatchNorm(hidden_dim)
        elif norm_type == "graph":
            self.norm = GraphNorm(hidden_dim)
        elif norm_type is None or norm_type == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch=None):
        residual = x

        x = self.conv(x, edge_index, edge_attr)

        if isinstance(self.norm, GraphNorm):
            x = self.norm(x, batch)
        else:
            x = self.norm(x)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x + residual

        return x

class NNConvRegressor(nn.Module):
    """NNConv graph-level regressor with Set2Set readout."""

    def __init__(
        self,
        node_dim=11,
        edge_dim=5,
        hidden_dim=64,
        num_layers=3,
        dropout=0.1,
        pooling="set2set",
        norm_type="batch",
        use_graph_features=False,
        graph_feat_dim=0,
    ):
        super().__init__()

        self.pooling = pooling
        self.use_graph_features = use_graph_features
        self.graph_feat_dim = graph_feat_dim

        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                NNConvBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_layers)
            ]
        )

        if pooling == "set2set":
            self.set2set = Set2Set(hidden_dim, processing_steps=3)
            pool_dim = 2 * hidden_dim
        elif pooling == "mean_add_max":
            self.set2set = None
            pool_dim = 3 * hidden_dim
        else:
            raise ValueError(f"Unsupported pooling for NNConvRegressor: {pooling}")

        if use_graph_features:
            pool_dim += graph_feat_dim

        self.head = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def pool(self, x, batch):
        if self.pooling == "set2set":
            return self.set2set(x, batch)

        if self.pooling == "mean_add_max":
            return torch.cat(
                [
                    global_mean_pool(x, batch),
                    global_add_pool(x, batch),
                    global_max_pool(x, batch),
                ],
                dim=1,
            )

        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = self.node_encoder(x)
        x = F.relu(x)

        for block in self.blocks:
            x = block(x, edge_index, edge_attr, batch=batch)

        x = self.pool(x, batch)

        if self.use_graph_features:
            if not hasattr(data, "graph_features"):
                raise ValueError("use_graph_features=True but data.graph_features is missing.")
            x = torch.cat([x, data.graph_features], dim=1)

        out = self.head(x)

        return out.view(-1)

class SchNetRegressor(nn.Module):
    """SchNet regressor for QM9 dipole moment prediction.

    Uses atomic numbers z and 3D coordinates pos.
    Ignores edge_index/edge_attr because SchNet builds distance-based
    continuous-filter interactions internally.
    """

    def __init__(
        self,
        atomic_number_col=5,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        max_num_neighbors=32,
        readout="add",
        dipole=False,
        dropout=0.0,
    ):
        super().__init__()

        self.atomic_number_col = atomic_number_col
        self.dropout = dropout
        self.dipole = dipole

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
            dipole=dipole,
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

        out = self.schnet(z=z, pos=pos, batch=batch)

        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        return out.view(-1)
