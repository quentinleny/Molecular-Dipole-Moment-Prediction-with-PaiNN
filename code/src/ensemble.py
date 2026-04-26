# ensemble.py
# Weighted ensemble of saved HW4 model checkpoints.

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import config
import time
import dataset
import predict


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


def prepare_val_and_test_loaders(cfg):
    """Rebuild the same validation split and test loader for a checkpoint config."""
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
        graph_feature_scaler = dataset.fit_graph_feature_scaler(train_data)

        train_data = dataset.add_graph_features_to_graphs(
            train_data,
            graph_feature_scaler,
        )

        test_data = dataset.add_graph_features_to_graphs(
            test_data,
            graph_feature_scaler,
        )

    if cfg.get("use_dipole_features", False):
        dipole_feature_scaler = dataset.fit_dipole_feature_scaler(
            train_data,
            atomic_number_col=cfg.get("atomic_number_col", config.atomic_number_col),
        )

        train_data = dataset.add_dipole_features_to_graphs(
            train_data,
            scaler=dipole_feature_scaler,
            atomic_number_col=cfg.get("atomic_number_col", config.atomic_number_col),
        )

        test_data = dataset.add_dipole_features_to_graphs(
            test_data,
            scaler=dipole_feature_scaler,
            atomic_number_col=cfg.get("atomic_number_col", config.atomic_number_col),
        )

    train_subset, val_subset, train_idx, val_idx = dataset.make_train_val_split(
        train_data,
        valid_frac=cfg.get("valid_frac", config.valid_frac),
        seed=cfg.get("seed", config.seed),
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
            out = model(batch).view(-1).detach().cpu()

            preds.extend(out.tolist())

            if hasattr(batch, "name"):
                names.extend(list(batch.name))

            if need_targets:
                targets.extend(batch.y.view(-1).detach().cpu().tolist())

    result = {
        "names": names,
        "preds": np.asarray(preds, dtype=float),
    }

    if need_targets:
        result["targets"] = np.asarray(targets, dtype=float)

    return result


def load_model_predictions(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    cfg = checkpoint.get("config", {})

    model = predict.build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loader, test_loader, sample_submission = prepare_val_and_test_loaders(cfg)

    val_result = predict_loader(
        model=model,
        loader=val_loader,
        device=device,
        need_targets=True,
    )

    test_result = predict_loader(
        model=model,
        loader=test_loader,
        device=device,
        need_targets=False,
    )

    print(f"Loaded {checkpoint_path}")
    print(f"  checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"  checkpoint val MAE: {checkpoint.get('val_mae')}")
    print(f"  val preds: {val_result['preds'].shape}")
    print(f"  test preds: {test_result['preds'].shape}")

    return {
        "checkpoint": checkpoint,
        "cfg": cfg,
        "val": val_result,
        "test": test_result,
        "sample_submission": sample_submission,
    }


def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def sweep_weights(pred_a, pred_b, target):
    rows = []

    for w in np.linspace(0.0, 1.0, 101):
        pred = w * pred_a + (1.0 - w) * pred_b
        rows.append(
            {
                "w_a": float(w),
                "w_b": float(1.0 - w),
                "mae": mae(pred, target),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("mae").reset_index(drop=True)

    return df


def next_ensemble_path():
    config.submission_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        path = config.submission_dir / f"ensemble_submission{i}.csv"
        if not path.exists():
            return path
        i += 1


def main():
    start_time = time.time()
    config.make_dirs()
    device = get_device()

    schnet_path = config.checkpoint_dir / "schnet_mu_dipole_best.pt"
    dimenet_path = config.checkpoint_dir / "dimenetpp_mu_best.pt"

    print("Device:", device)
    print("SchNet checkpoint:", schnet_path)
    print("DimeNet++ checkpoint:", dimenet_path)

    schnet = load_model_predictions(schnet_path, device)
    dimenet = load_model_predictions(dimenet_path, device)

    # Validation split should be identical by construction.
    y = schnet["val"]["targets"]

    schnet_val = schnet["val"]["preds"]
    dimenet_val = dimenet["val"]["preds"]

    print("\nSingle-model validation MAE:")
    print("  SchNet-dipole:", mae(schnet_val, y))
    print("  DimeNet++:     ", mae(dimenet_val, y))

    sweep = sweep_weights(
        pred_a=schnet_val,
        pred_b=dimenet_val,
        target=y,
    )

    sweep_path = config.diagnostic_dir / "ensemble_weight_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    best = sweep.iloc[0]
    w_schnet = float(best["w_a"])
    w_dimenet = float(best["w_b"])

    print("\nBest ensemble weight:")
    print(best)
    print("Saved sweep:", sweep_path)

    elapsed = time.time() - start_time
    print(f"\nTotal ensemble time: {elapsed / 60:.2f} min")

    schnet_test = schnet["test"]["preds"]
    dimenet_test = dimenet["test"]["preds"]
    test_names = schnet["test"]["names"]

    ensemble_test = w_schnet * schnet_test + w_dimenet * dimenet_test

    pred_map = dict(zip(test_names, ensemble_test))

    sample_submission = schnet["sample_submission"]
    submission = dataset.make_prediction_frame(sample_submission, pred_map)

    output_path = next_ensemble_path()
    submission.to_csv(output_path, index=False)

    print("\nSaved ensemble submission:", output_path)
    print(submission["labels"].describe())



if __name__ == "__main__":
    main()
