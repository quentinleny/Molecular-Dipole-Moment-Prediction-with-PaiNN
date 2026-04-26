# predict.py
# Load trained HW4 GNN checkpoint and create Kaggle submission CSV.

from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

import config
import dataset
import models
import utils

def get_device():
    """Use CUDA if available; otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def next_submission_path(submission_dir, prefix="submission", suffix=".csv", start_idx=2):
    """Return next available submission path: submission2.csv, submission3.csv, ..."""
    submission_dir = Path(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    idx = start_idx
    while True:
        path = submission_dir / f"{prefix}{idx}{suffix}"
        if not path.exists():
            return path
        idx += 1

def build_model_from_checkpoint(checkpoint):
    """Instantiate model using checkpoint config when available."""
    cfg = checkpoint.get("config", {})

    model_name = cfg.get("model_name", config.model_name)
    use_distances = cfg.get("use_distances", config.use_distances)
    edge_dim = cfg.get(
        "edge_dim_aug" if use_distances else "edge_dim_raw",
        config.edge_dim_aug if use_distances else config.edge_dim_raw,
    )

    if model_name.startswith("gine"):
        return models.GINERegressor(
            node_dim=cfg.get("node_dim", config.node_dim),
            edge_dim=edge_dim,
            hidden_dim=cfg.get("hidden_dim", config.hidden_dim),
            num_layers=cfg.get("num_layers", config.num_layers),
            dropout=cfg.get("dropout", config.dropout),
            pooling=cfg.get("pooling", config.pooling),
            norm_type=cfg.get("norm_type", config.norm_type),
            use_graph_features=cfg.get("use_graph_features", config.use_graph_features),
            graph_feat_dim=cfg.get("graph_feat_dim", config.graph_feat_dim),
        )

    if model_name.startswith("nnconv"):
        return models.NNConvRegressor(
            node_dim=cfg.get("node_dim", config.node_dim),
            edge_dim=edge_dim,
            hidden_dim=cfg.get("hidden_dim", config.hidden_dim),
            num_layers=cfg.get("num_layers", config.num_layers),
            dropout=cfg.get("dropout", config.dropout),
            pooling=cfg.get("pooling", config.pooling),
            norm_type=cfg.get("norm_type", config.norm_type),
            use_graph_features=cfg.get("use_graph_features", config.use_graph_features),
            graph_feat_dim=cfg.get("graph_feat_dim", config.graph_feat_dim),
        )

    if model_name.startswith("painn"):
        return models.PaiNNRegressor(
            atomic_number_col=cfg.get("atomic_number_col", config.atomic_number_col),
            hidden_dim=cfg.get("painn_hidden_dim", config.painn_hidden_dim),
            num_layers=cfg.get("painn_num_layers", config.painn_num_layers),
            num_radial=cfg.get("painn_num_radial", config.painn_num_radial),
            cutoff=cfg.get("painn_cutoff", config.painn_cutoff),
            max_num_neighbors=cfg.get(
                "painn_max_num_neighbors",
                config.painn_max_num_neighbors,
            ),
            center_mode=cfg.get("painn_center_mode", config.painn_center_mode),
        )

    if model_name.startswith("schnet"):
        return models.SchNetRegressor(
            atomic_number_col=cfg.get("atomic_number_col", config.atomic_number_col),
            hidden_channels=cfg.get("schnet_hidden_channels", config.schnet_hidden_channels),
            num_filters=cfg.get("schnet_num_filters", config.schnet_num_filters),
            num_interactions=cfg.get("schnet_num_interactions", config.schnet_num_interactions),
            num_gaussians=cfg.get("schnet_num_gaussians", config.schnet_num_gaussians),
            cutoff=cfg.get("schnet_cutoff", config.schnet_cutoff),
            max_num_neighbors=cfg.get(
                "schnet_max_num_neighbors",
                config.schnet_max_num_neighbors,
            ),
            readout=cfg.get("schnet_readout", config.schnet_readout),
            dipole=cfg.get("schnet_dipole", config.schnet_dipole),
            dropout=cfg.get("dropout", config.dropout),
        )

    raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")

def load_test_loader():
    """Load and preprocess test data for prediction."""
    train_data, test_data, sample_submission = dataset.load_datasets(config.data_dir)

    dataset.validate_graphs(train_data, n=10)
    dataset.validate_graphs(test_data, n=10)
    dataset.check_submission_ids(test_data, sample_submission)

    train_data = dataset.preprocess_graphs(
        train_data,
        use_distances=config.use_distances,
    )

    test_data = dataset.preprocess_graphs(
        test_data,
        use_distances=config.use_distances,
    )

    if config.use_graph_features:
        graph_feature_scaler = dataset.fit_graph_feature_scaler(train_data)

        test_data = dataset.add_graph_features_to_graphs(
            test_data,
            graph_feature_scaler,
        )

    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return test_loader, sample_submission

def predict(checkpoint_path=None, submission_path=None):
    """Create Kaggle submission from saved checkpoint."""
    config.make_dirs()

    device = get_device()

    if checkpoint_path is None:
        checkpoint_path = config.checkpoint_dir / config.checkpoint_name
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint val MAE:", checkpoint.get("val_mae"))

    test_loader, sample_submission = load_test_loader()

    pred_map = utils.predict(
        model=model,
        loader=test_loader,
        device=device,
    )

    summary = utils.prediction_summary(pred_map)
    print("Prediction summary:", summary)

    submission = dataset.make_prediction_frame(sample_submission, pred_map)

    if submission_path is None:
        submission_path = next_submission_path(config.submission_dir, start_idx=2)
    else:
        submission_path = Path(submission_path)

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)

    print("Saved submission:", submission_path)

    return {
        "submission_path": submission_path,
        "checkpoint_path": checkpoint_path,
        "prediction_summary": summary,
    }

if __name__ == "__main__":
    predict()
