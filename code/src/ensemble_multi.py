# ensemble_multi.py
# Multi-model weighted ensemble with validation-based weight search.

from pathlib import Path
import itertools
import time
import models

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import config
import dataset
import predict


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_val_and_test_loaders(cfg):
    train_data, test_data, sample_submission = dataset.load_datasets(config.data_dir)

    train_data = dataset.preprocess_graphs(
        train_data,
        use_distances=cfg.get("use_distances", config.use_distances),
    )
    test_data = dataset.preprocess_graphs(
        test_data,
        use_distances=cfg.get("use_distances", config.use_distances),
    )

    if cfg.get("use_graph_features", False):
        scaler = dataset.fit_graph_feature_scaler(train_data)
        train_data = dataset.add_graph_features_to_graphs(train_data, scaler)
        test_data = dataset.add_graph_features_to_graphs(test_data, scaler)

    if cfg.get("use_dipole_features", False):
        scaler = dataset.fit_dipole_feature_scaler(
            train_data,
            atomic_number_col=cfg.get("atomic_number_col", config.atomic_number_col),
        )
        train_data = dataset.add_dipole_features_to_graphs(
            train_data,
            scaler=scaler,
            atomic_number_col=cfg.get("atomic_number_col", config.atomic_number_col),
        )
        test_data = dataset.add_dipole_features_to_graphs(
            test_data,
            scaler=scaler,
            atomic_number_col=cfg.get("atomic_number_col", config.atomic_number_col),
        )

    train_subset, val_subset, train_idx, val_idx = dataset.make_train_val_split(
        train_data,
        valid_frac=cfg.get("valid_frac", config.valid_frac),
        seed=cfg.get("split_seed", config.split_seed),
        stratified=cfg.get("use_stratified_split", config.use_stratified_split),
        num_strat_bins=cfg.get("num_strat_bins", config.num_strat_bins),
    )

    batch_size = cfg.get("batch_size", config.batch_size)
    num_workers = cfg.get("num_workers", config.num_workers)

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

    return val_loader, test_loader, sample_submission


def predict_loader(model, loader, device, need_targets=False):
    model.eval()

    names = []
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1).detach().cpu().numpy()
            preds.append(out)

            if hasattr(batch, "name"):
                names.extend(list(batch.name))

            if need_targets:
                targets.append(batch.y.view(-1).detach().cpu().numpy())

    result = {
        "names": names,
        "preds": np.concatenate(preds),
    }

    if need_targets:
        result["targets"] = np.concatenate(targets)

    return result

def build_model_from_cfg(cfg):
    model_name = cfg.get("model_name", "")

    if "painn" in model_name:
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
            dropout=cfg.get("dropout", config.dropout),
        )

    if "schnet" in model_name:
        return models.SchNetDipoleRegressor(
            atomic_number_col=cfg.get("atomic_number_col", config.atomic_number_col),
            hidden_channels=cfg.get(
                "schnet_hidden_channels",
                config.schnet_hidden_channels,
            ),
            num_filters=cfg.get("schnet_num_filters", config.schnet_num_filters),
            num_interactions=cfg.get(
                "schnet_num_interactions",
                config.schnet_num_interactions,
            ),
            num_gaussians=cfg.get(
                "schnet_num_gaussians",
                config.schnet_num_gaussians,
            ),
            cutoff=cfg.get("schnet_cutoff", config.schnet_cutoff),
            max_num_neighbors=cfg.get(
                "schnet_max_num_neighbors",
                config.schnet_max_num_neighbors,
            ),
            readout=cfg.get("schnet_readout", config.schnet_readout),
            dipole=cfg.get("schnet_dipole", config.schnet_dipole),
        )

    raise ValueError(f"Unknown model_name in checkpoint config: {model_name}")

def load_model_predictions(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"SKIP missing checkpoint: {checkpoint_path.name}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    model = build_model_from_cfg(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loader, test_loader, sample_submission = prepare_val_and_test_loaders(cfg)

    val_result = predict_loader(model, val_loader, device, need_targets=True)
    test_result = predict_loader(model, test_loader, device, need_targets=False)

    val_mae = np.mean(np.abs(val_result["preds"] - val_result["targets"]))

    print(f"Loaded: {checkpoint_path.name}")
    print(f"  model_name: {cfg.get('model_name')}")
    print(f"  seed: {cfg.get('seed')} | split_seed: {cfg.get('split_seed')}")
    print(f"  checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"  checkpoint val MAE: {checkpoint.get('val_mae', checkpoint.get('best_val_mae'))}")
    print(f"  recomputed val MAE: {val_mae:.6f}")

    return {
        "path": checkpoint_path,
        "checkpoint": checkpoint,
        "cfg": cfg,
        "val": val_result,
        "test": test_result,
        "sample_submission": sample_submission,
    }


def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def simplex_weights(n_models, step=0.05):
    units = int(round(1.0 / step))

    for counts in itertools.product(range(units + 1), repeat=n_models):
        if sum(counts) == units:
            yield np.array(counts, dtype=float) / units


def sweep_simplex(val_preds, target, step=0.05):
    rows = []
    pred_matrix = np.stack(val_preds, axis=0)

    for w in simplex_weights(len(val_preds), step=step):
        pred = np.sum(w[:, None] * pred_matrix, axis=0)
        row = {f"w_{i}": float(w[i]) for i in range(len(w))}
        row["mae"] = mae(pred, target)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def next_ensemble_path():
    config.submission_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        path = config.submission_dir / f"multi_ensemble_submission{i}.csv"
        if not path.exists():
            return path
        i += 1


def main():
    start_time = time.time()
    config.make_dirs()
    device = get_device()

    checkpoint_paths = [
        # Best PaiNN checkpoints
        config.checkpoint_dir / "painn_h128_l6_rbf64_cutoff10_seed60_split60_best.pt",
        config.checkpoint_dir / "painn_h128_l6_rbf64_cutoff10_seed60_split60_1_best.pt",
        config.checkpoint_dir / "painn_h128_l6_rbf64_cutoff10_seed60_split60_3_best.pt",

        # Best SchNet checkpoints
        config.checkpoint_dir / "schnet_mu_dipole_h192_f192_i6_seed60_split60_best.pt",
        config.checkpoint_dir / "schnet_mu_dipole_seed36_split60_best.pt",
        config.checkpoint_dir / "schnet_mu_dipole_seed12_split60_best.pt",
    ]

    print("Device:", device)

    models_loaded = []

    for path in checkpoint_paths:
        result = load_model_predictions(path, device)
        if result is not None:
            models_loaded.append(result)

    if len(models_loaded) < 2:
        raise ValueError("Need at least two checkpoints for ensemble.")

    target = models_loaded[0]["val"]["targets"]

    print("\nSingle-model validation MAEs:")
    val_preds = []

    for i, item in enumerate(models_loaded):
        pred = item["val"]["preds"]
        val_preds.append(pred)
        print(f"{i}: {item['path'].name} | MAE={mae(pred, target):.6f}")

    print("\nValidation prediction correlations:")
    pred_matrix = np.stack(val_preds, axis=0)
    corr = np.corrcoef(pred_matrix)

    for i in range(len(models_loaded)):
        row = " ".join([f"{corr[i, j]:.4f}" for j in range(len(models_loaded))])
        print(f"{i}: {row}")

    sweep = sweep_simplex(val_preds, target, step=0.05)

    sweep_path = config.diagnostic_dir / "multi_ensemble_weight_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    best = sweep.iloc[0]
    weights = np.array([best[f"w_{i}"] for i in range(len(models_loaded))], dtype=float)

    print("\nBest ensemble:")
    print(best)
    print("Saved sweep:", sweep_path)

    test_preds = np.stack([item["test"]["preds"] for item in models_loaded], axis=0)
    ensemble_test = np.sum(weights[:, None] * test_preds, axis=0)

    test_names = models_loaded[0]["test"]["names"]
    pred_map = dict(zip(test_names, ensemble_test))

    sample_submission = models_loaded[0]["sample_submission"]
    submission = dataset.make_prediction_frame(sample_submission, pred_map)

    output_path = next_ensemble_path()
    submission.to_csv(output_path, index=False)

    print("\nModel weights:")
    for i, item in enumerate(models_loaded):
        print(f"{weights[i]:.3f} | {item['path'].name}")

    print("\nSaved ensemble submission:", output_path)
    print(submission["labels"].describe())

    elapsed = time.time() - start_time
    print(f"\nTotal ensemble time: {elapsed / 60:.2f} min")


if __name__ == "__main__":
    main()
