# dataset.py
# Data loading, preprocessing, splitting, and PyG loader utilities for HW4.

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import is_undirected

def load_datasets(data_dir):
    """Load train/test PyG Data lists and sample submission."""
    data_dir = Path(data_dir)

    train_path = data_dir / "train.pt"
    test_path = data_dir / "test.pt"
    submission_path = data_dir / "sample_submission.csv"

    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    sample_submission = pd.read_csv(submission_path)

    return train_data, test_data, sample_submission

def validate_graphs(graphs, n=10):
    """Validate a small sample of PyG Data objects."""
    n_check = min(n, len(graphs))

    for i in range(n_check):
        graph = graphs[i]
        graph.validate(raise_on_error=True)

        if graph.x is None:
            raise ValueError(f"Graph {i} is missing x.")
        if graph.edge_index is None:
            raise ValueError(f"Graph {i} is missing edge_index.")
        if graph.edge_attr is None:
            raise ValueError(f"Graph {i} is missing edge_attr.")
        if graph.pos is None:
            raise ValueError(f"Graph {i} is missing pos.")
        if not hasattr(graph, "name"):
            raise ValueError(f"Graph {i} is missing name.")

    return True

def check_submission_ids(test_data, sample_submission):
    """Check that sample submission IDs match test graph names."""
    test_names = {graph.name for graph in test_data}
    submission_names = set(sample_submission["Idx"])

    if test_names != submission_names:
        missing_in_submission = test_names - submission_names
        missing_in_test = submission_names - test_names

        raise ValueError(
            "Submission IDs do not match test graph names. "
            f"Missing in submission: {len(missing_in_submission)}. "
            f"Missing in test: {len(missing_in_test)}."
        )

    return True

def summarize_undirected_status(graphs, n=100):
    """Return how many sampled graphs have undirected edge_index."""
    n_check = min(n, len(graphs))
    flags = []

    for i in range(n_check):
        graph = graphs[i]
        flags.append(bool(is_undirected(graph.edge_index, num_nodes=graph.num_nodes)))

    return {
        "n_checked": n_check,
        "n_undirected": int(sum(flags)),
        "all_checked_undirected": bool(all(flags)),
    }

def add_edge_distances(graph):
    """Append bond distances to edge_attr once.

    Original edge_attr: [num_edges, 4]
    Augmented edge_attr: [num_edges, 5]
    """
    if graph.edge_attr is None:
        raise ValueError("Cannot add distances because edge_attr is missing.")
    if graph.pos is None:
        raise ValueError("Cannot add distances because pos is missing.")

    edge_dim = graph.edge_attr.size(1)

    if edge_dim == 4:
        row, col = graph.edge_index
        dist = torch.norm(graph.pos[row] - graph.pos[col], dim=1, keepdim=True)
        graph.edge_attr = torch.cat([graph.edge_attr, dist], dim=1)
    elif edge_dim == 5:
        pass
    else:
        raise ValueError(f"Unexpected edge_attr dimension: {edge_dim}")

    return graph

def preprocess_graph(graph, use_distances=True):
    """Cast fields and optionally append edge distances."""
    graph.x = graph.x.float()
    graph.edge_attr = graph.edge_attr.float()
    graph.pos = graph.pos.float()
    graph.y = torch.as_tensor(graph.y, dtype=torch.float).view(1)

    if use_distances:
        graph = add_edge_distances(graph)

    return graph

def preprocess_graphs(graphs, use_distances=True):
    """Preprocess all graphs in-place and return the same list."""
    return [preprocess_graph(graph, use_distances=use_distances) for graph in graphs]

def get_targets(graphs):
    """Return graph-level targets as a NumPy array."""
    return np.array([float(graph.y.view(-1)[0]) for graph in graphs], dtype=float)

def raw_graph_features(graph):
    """Return raw graph-level features: num_nodes, num_edges, edge/node ratio."""
    num_nodes = float(graph.num_nodes)
    num_edges = float(graph.edge_index.size(1))
    edge_node_ratio = num_edges / max(num_nodes, 1.0)

    return torch.tensor(
        [num_nodes, num_edges, edge_node_ratio],
        dtype=torch.float,
    )

def fit_graph_feature_scaler(graphs):
    """Compute train-set mean/std for graph-level features."""
    feats = torch.stack([raw_graph_features(graph) for graph in graphs], dim=0)

    mean = feats.mean(dim=0)
    std = feats.std(dim=0)

    std = torch.where(std > 0, std, torch.ones_like(std))

    return {
        "mean": mean,
        "std": std,
    }

def add_graph_features(graph, scaler):
    """Attach standardized graph-level features as shape [1, 3].

    Shape [1, 3] is intentional so PyG batching gives [batch_size, 3].
    """
    feat = raw_graph_features(graph)

    feat = (feat - scaler["mean"]) / scaler["std"]

    graph.graph_features = feat.view(1, -1)

    return graph

def add_graph_features_to_graphs(graphs, scaler):
    """Attach standardized graph-level features to every graph."""
    return [add_graph_features(graph, scaler) for graph in graphs]

def make_stratification_bins(y, num_bins):
    """Create quantile bins for stratified graph-level splitting."""
    y = np.asarray(y, dtype=float)

    try:
        bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
        bins = np.asarray(bins)

        if pd.isna(bins).any():
            raise ValueError("qcut produced NaN bins.")

        unique_bins, counts = np.unique(bins, return_counts=True)

        if len(unique_bins) < 2:
            return None
        if np.min(counts) < 2:
            return None

        return bins
    except ValueError:
        return None

def make_train_val_split(
    train_data,
    valid_frac=0.10,
    seed=60,
    stratified=True,
    num_strat_bins=20,
):
    """Create graph-level train/validation split."""
    indices = np.arange(len(train_data))

    if stratified:
        y = get_targets(train_data)
        bins = make_stratification_bins(y, num_strat_bins)

        if bins is not None:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=valid_frac,
                random_state=seed,
                stratify=bins,
            )
        else:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=valid_frac,
                random_state=seed,
                shuffle=True,
            )
    else:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=valid_frac,
            random_state=seed,
            shuffle=True,
        )

    train_subset = [train_data[i] for i in train_idx]
    val_subset = [train_data[i] for i in val_idx]

    return train_subset, val_subset, train_idx, val_idx

def make_loaders(
    train_subset,
    val_subset,
    test_data,
    batch_size=64,
    num_workers=0,
):
    """Create PyG DataLoaders."""
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader

def make_prediction_frame(sample_submission, pred_map):
    """Create Kaggle submission DataFrame from graph-name prediction map."""
    submission = sample_submission.copy()
    submission["labels"] = submission["Idx"].map(pred_map)

    if submission["labels"].isna().any():
        n_missing = int(submission["labels"].isna().sum())
        raise ValueError(f"Missing predictions for {n_missing} test graphs.")

    return submission

def describe_loader_batch(loader):
    """Return a lightweight summary of the first batch for debugging."""
    batch = next(iter(loader))

    summary = {
        "num_graphs": batch.num_graphs,
        "x_shape": tuple(batch.x.shape),
        "edge_index_shape": tuple(batch.edge_index.shape),
        "edge_attr_shape": tuple(batch.edge_attr.shape),
        "y_shape": tuple(batch.y.shape),
        "batch_shape": tuple(batch.batch.shape),
        "has_name": hasattr(batch, "name"),
    }

    if hasattr(batch, "name"):
        summary["first_names"] = list(batch.name[:5])

    return summary

def atom_scalar_values(z, values):
    """Map atomic number tensor to scalar property values."""
    out = torch.zeros_like(z, dtype=torch.float)

    for atom_num, value in values.items():
        out[z == atom_num] = float(value)

    return out

def dipole_graph_features(graph, atomic_number_col=5):
    """Return physics-inspired dipole features as shape [1, 16].

    Features:
    - num_nodes
    - atomic-number moment norm
    - electronegativity moment norm
    - mass moment norm
    - atom-type centered-position moment norms for H,C,N,O,F
    - pairwise distance mean/std/min/max
    - coordinate spread norms
    - approximate charge-imbalance proxy
    """
    x = graph.x.float()
    pos = graph.pos.float()
    z = x[:, atomic_number_col].long()

    center = pos.mean(dim=0, keepdim=True)
    r = pos - center

    num_nodes = torch.tensor([float(pos.size(0))], dtype=torch.float, device=pos.device)

    electronegativity = {
        1: 2.20,
        6: 2.55,
        7: 3.04,
        8: 3.44,
        9: 3.98,
    }

    atomic_mass = {
        1: 1.008,
        6: 12.011,
        7: 14.007,
        8: 15.999,
        9: 18.998,
    }

    chi = atom_scalar_values(z, electronegativity).view(-1, 1)
    mass = atom_scalar_values(z, atomic_mass).view(-1, 1)
    z_float = z.float().view(-1, 1)

    z_moment = torch.norm(torch.sum(z_float * r, dim=0)).view(1)
    chi_moment = torch.norm(torch.sum(chi * r, dim=0)).view(1)
    mass_moment = torch.norm(torch.sum(mass * r, dim=0)).view(1)

    type_moments = []
    for atom_num in [1, 6, 7, 8, 9]:
        mask = z == atom_num
        if mask.any():
            type_moment = torch.norm(torch.sum(r[mask], dim=0)).view(1)
        else:
            type_moment = torch.zeros(1, dtype=torch.float, device=pos.device)
        type_moments.append(type_moment)

    if pos.size(0) >= 2:
        d = torch.pdist(pos, p=2)
        d_mean = d.mean().view(1)
        d_std = d.std(unbiased=False).view(1)
        d_min = d.min().view(1)
        d_max = d.max().view(1)
    else:
        d_mean = torch.zeros(1, dtype=torch.float, device=pos.device)
        d_std = torch.zeros(1, dtype=torch.float, device=pos.device)
        d_min = torch.zeros(1, dtype=torch.float, device=pos.device)
        d_max = torch.zeros(1, dtype=torch.float, device=pos.device)

    coord_std = r.std(dim=0, unbiased=False)
    coord_spread_norm = torch.norm(coord_std).view(1)

    centered_chi = chi - chi.mean()
    charge_proxy = torch.norm(torch.sum(centered_chi * r, dim=0)).view(1)

    feat = torch.cat(
        [
            num_nodes,
            z_moment,
            chi_moment,
            mass_moment,
            *type_moments,
            d_mean,
            d_std,
            d_min,
            d_max,
            coord_std,
            coord_spread_norm,
            charge_proxy,
        ],
        dim=0,
    )

    return feat.view(1, -1)

def fit_dipole_feature_scaler(graphs, atomic_number_col=5):
    """Compute train-set mean/std for dipole graph features."""
    feats = torch.cat(
        [
            dipole_graph_features(graph, atomic_number_col=atomic_number_col)
            for graph in graphs
        ],
        dim=0,
    )

    mean = feats.mean(dim=0)
    std = feats.std(dim=0, unbiased=False)
    std = torch.where(std > 0, std, torch.ones_like(std))

    return {
        "mean": mean,
        "std": std,
    }

def add_dipole_features(graph, scaler, atomic_number_col=5):
    """Attach standardized dipole features as graph.dipole_features."""
    feat = dipole_graph_features(graph, atomic_number_col=atomic_number_col)
    feat = (feat - scaler["mean"]) / scaler["std"]
    graph.dipole_features = feat
    return graph

def add_dipole_features_to_graphs(graphs, scaler, atomic_number_col=5):
    """Attach standardized dipole features to every graph."""
    return [
        add_dipole_features(
            graph,
            scaler=scaler,
            atomic_number_col=atomic_number_col,
        )
        for graph in graphs
    ]
