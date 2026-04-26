# train.py
# Train HW4 GNN model and save best checkpoint by validation MAE.

import time
from pathlib import Path

import torch

import config
import dataset
import models
import utils

def get_device():
    """Use CUDA if available; otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    """Instantiate the configured model."""
    edge_dim = config.edge_dim_aug if config.use_distances else config.edge_dim_raw

    if config.model_name.startswith("gine"):
        return models.GINERegressor(
            node_dim=config.node_dim,
            edge_dim=edge_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            pooling=config.pooling,
            norm_type=config.norm_type,
            use_graph_features=config.use_graph_features,
            graph_feat_dim=config.graph_feat_dim,
        )

    if config.model_name.startswith("nnconv"):
        return models.NNConvRegressor(
            node_dim=config.node_dim,
            edge_dim=edge_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            pooling=config.pooling,
            norm_type=config.norm_type,
            use_graph_features=config.use_graph_features,
            graph_feat_dim=config.graph_feat_dim,
        )

    if config.model_name.startswith("painn"):
        return models.PaiNNRegressor(
            atomic_number_col=config.atomic_number_col,
            hidden_dim=config.painn_hidden_dim,
            num_layers=config.painn_num_layers,
            num_radial=config.painn_num_radial,
            cutoff=config.painn_cutoff,
            max_num_neighbors=config.painn_max_num_neighbors,
            center_mode=config.painn_center_mode,
        )

    if config.model_name.startswith("schnet"):
        return models.SchNetRegressor(
            atomic_number_col=config.atomic_number_col,
            hidden_channels=config.schnet_hidden_channels,
            num_filters=config.schnet_num_filters,
            num_interactions=config.schnet_num_interactions,
            num_gaussians=config.schnet_num_gaussians,
            cutoff=config.schnet_cutoff,
            max_num_neighbors=config.schnet_max_num_neighbors,
            readout=config.schnet_readout,
            dipole=config.schnet_dipole,
            dropout=config.dropout,
        )

    raise ValueError(f"Unsupported model_name: {config.model_name}")

def prepare_data():
    """Load, validate, preprocess, split, and build loaders."""
    train_data, test_data, sample_submission = dataset.load_datasets(config.data_dir)

    dataset.validate_graphs(train_data, n=10)
    dataset.validate_graphs(test_data, n=10)
    dataset.check_submission_ids(test_data, sample_submission)

    undirected_status = dataset.summarize_undirected_status(train_data, n=100)
    print("Undirected status:", undirected_status)

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

        train_data = dataset.add_graph_features_to_graphs(
            train_data,
            graph_feature_scaler,
        )

        test_data = dataset.add_graph_features_to_graphs(
            test_data,
            graph_feature_scaler,
        )

    split_seed = getattr(config, "split_seed", config.seed)

    train_subset, val_subset, train_idx, val_idx = dataset.make_train_val_split(
        train_data,
        valid_frac=config.valid_frac,
        seed=split_seed,
        stratified=config.use_stratified_split,
        num_strat_bins=config.num_strat_bins,
    )

    train_loader, val_loader, test_loader = dataset.make_loaders(
        train_subset,
        val_subset,
        test_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    print("Split sizes:", len(train_subset), len(val_subset), len(test_data))
    print("Train batch:", dataset.describe_loader_batch(train_loader))
    print("Val batch:", dataset.describe_loader_batch(val_loader))

    return train_loader, val_loader, test_loader, train_idx, val_idx

def train():
    """Run full training."""
    config.make_dirs()
    utils.set_seed(config.seed)

    device = get_device()
    print("Device:", device)
    print("Project root:", config.project_root)
    print("Data dir:", config.data_dir)
    print("seed:", config.seed)
    print("split_seed:", config.split_seed)
    print("PaiNN hidden_dim:", config.painn_hidden_dim)
    print("PaiNN num_layers:", config.painn_num_layers)
    print("PaiNN num_radial:", config.painn_num_radial)
    print("PaiNN cutoff:", config.painn_cutoff)
    print("LR:", config.lr)
    print("Batch size:", config.batch_size)
    print("Dropout:", config.dropout)

    train_loader, val_loader, test_loader, train_idx, val_idx = prepare_data()

    model = build_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    loss_fn = utils.get_loss_fn(config.loss_name)

    scheduler = None
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )

    checkpoint_path = config.checkpoint_dir / config.checkpoint_name
    log_path = config.log_dir / config.log_name

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    rows = []

    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        train_loss, train_mae = utils.train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )

        val_mae, val_preds, val_targets = utils.evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        if scheduler is not None:
            scheduler.step(val_mae)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        improved = val_mae < best_val_mae

        if improved:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0

            utils.save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_mae=val_mae,
                config_dict=config.as_dict(),
            )
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
            "improved": improved,
        }
        rows.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"train_mae={train_mae:.6f} | "
            f"val_mae={val_mae:.6f} | "
            f"best={best_val_mae:.6f} @ {best_epoch:03d} | "
            f"lr={current_lr:.2e} | "
            f"time={epoch_time:.1f}s"
        )

        utils.save_log(log_path, rows)

        if epochs_without_improvement >= config.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best val MAE {best_val_mae:.6f} at epoch {best_epoch}."
            )
            break

    total_time = time.time() - start_time

    print("Training complete.")
    print(f"Best val MAE: {best_val_mae:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log: {log_path}")
    print(f"Total time: {total_time / 60:.2f} min")

    return {
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
        "log_path": log_path,
    }

if __name__ == "__main__":
    train()
