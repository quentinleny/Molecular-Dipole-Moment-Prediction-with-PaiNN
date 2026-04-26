# config.py
# Central configuration for HW4 GNN experiments.

from pathlib import Path


def find_project_root(start=None):
    path = Path.cwd().resolve() if start is None else Path(start).resolve()

    for candidate in [path] + list(path.parents):
        if (candidate / "data" / "train.pt").exists() and (candidate / "code" / "src").exists():
            return candidate

    raise FileNotFoundError("Could not find HW4 project root containing data/train.pt and code/src.")


project_root = find_project_root()

data_dir = project_root / "data"

code_dir = project_root / "code"
src_dir = code_dir / "src"
notebook_dir = code_dir / "notebooks"

output_dir = project_root / "outputs"
checkpoint_dir = output_dir / "checkpoints"
log_dir = output_dir / "logs"
submission_dir = output_dir / "submissions"
diagnostic_dir = output_dir / "diagnostics"
eda_dir = output_dir / "eda"

# Reproducibility
seed = 60
split_seed = 60
deterministic = False

# Data
node_dim = 11
edge_dim_raw = 4
edge_dim_aug = 5
use_distances = False

use_atom_onehot = False
atomic_number_col = 5
atom_values = [1, 6, 7, 8, 9]

# Split
valid_frac = 0.10
use_stratified_split = True
num_strat_bins = 20

# Model
model_name = "painn_h128_l6_rbf64_cutoff10_seed60_split60"
hidden_dim = 128
num_layers = 4
pooling = "painn_dipole"
norm_type = "none"

use_graph_features = False
graph_feat_dim = 0

# SchNet
schnet_hidden_channels = 128
schnet_num_filters = 128
schnet_num_interactions = 6
schnet_num_gaussians = 50
schnet_cutoff = 10.0
schnet_max_num_neighbors = 32
schnet_readout = "add"
schnet_dipole = True

# PaiNN
painn_hidden_dim = 128
painn_num_layers = 4
painn_num_radial = 64
painn_cutoff = 7.5
painn_max_num_neighbors = 32
painn_center_mode = "mass"

# painn_readout = "dipole_norm"
# painn_use_charge_dipoles = True
# painn_use_local_dipoles = True
# painn_enforce_charge_neutrality = True
# painn_center_positions = "mass"
# painn_epsilon = 1e-8
# painn_residual_scale = 1.0
# painn_vector_gate = True
# painn_use_layer_norm = True

# Optional physics/diagnostic features
use_dipole_features = False
dipole_feat_dim = 18

# Training
sweep_run = 5
epochs = 350
patience = 45
lr = 1e-3
batch_size = 64
dropout = 0.0
num_workers = 0
weight_decay = 0
loss_name = "l1"
grad_clip_norm = 10.0

# Scheduler
use_scheduler = True
scheduler_factor = 0.5
scheduler_patience = 15
min_lr = 1e-6

# Target handling
use_target_standardization = False

# Diagnostics
diagnostic_every = 25
save_val_predictions_every = 0

run_equivariance_check = False
equivariance_check_every = 25
equivariance_check_batches = 1

run_eda = False
eda_num_graphs = 20000
eda_pairwise_distance_sample = 2000

# Output names
checkpoint_name = f"{model_name}_{sweep_run}_best.pt"
log_name = f"{model_name}_{sweep_run}_train_log.csv"
submission_name = f"{model_name}_{sweep_run}_submission.csv"
diagnostic_name = f"{model_name}_{sweep_run}_diagnostics.csv"
val_prediction_name = f"{model_name}_{sweep_run}_val_predictions.csv"
eda_name = "data_eda_summary.csv"


def make_dirs():
    """Create output directories."""
    for path in [
        output_dir,
        checkpoint_dir,
        log_dir,
        submission_dir,
        diagnostic_dir,
        eda_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def as_dict():
    """Return config as a plain dictionary for checkpoint metadata."""
    return {
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "log_dir": str(log_dir),
        "submission_dir": str(submission_dir),
        "diagnostic_dir": str(diagnostic_dir),
        "eda_dir": str(eda_dir),

        "seed": seed,
        "split_seed": split_seed,
        "deterministic": deterministic,

        "node_dim": node_dim,
        "edge_dim_raw": edge_dim_raw,
        "edge_dim_aug": edge_dim_aug,
        "use_distances": use_distances,

        "use_atom_onehot": use_atom_onehot,
        "atomic_number_col": atomic_number_col,
        "atom_values": atom_values,

        "valid_frac": valid_frac,
        "use_stratified_split": use_stratified_split,
        "num_strat_bins": num_strat_bins,

        "batch_size": batch_size,
        "num_workers": num_workers,

        "model_name": model_name,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "pooling": pooling,
        "norm_type": norm_type,

        "use_graph_features": use_graph_features,
        "graph_feat_dim": graph_feat_dim,

        "schnet_hidden_channels": schnet_hidden_channels,
        "schnet_num_filters": schnet_num_filters,
        "schnet_num_interactions": schnet_num_interactions,
        "schnet_num_gaussians": schnet_num_gaussians,
        "schnet_cutoff": schnet_cutoff,
        "schnet_max_num_neighbors": schnet_max_num_neighbors,
        "schnet_readout": schnet_readout,
        "schnet_dipole": schnet_dipole,

        "painn_hidden_dim": painn_hidden_dim,
        "painn_num_layers": painn_num_layers,
        "painn_num_radial": painn_num_radial,
        "painn_cutoff": painn_cutoff,
        "painn_max_num_neighbors": painn_max_num_neighbors,
        "painn_center_mode": painn_center_mode,

        "use_dipole_features": use_dipole_features,
        "dipole_feat_dim": dipole_feat_dim,

        "epochs": epochs,
        "patience": patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "loss_name": loss_name,
        "grad_clip_norm": grad_clip_norm,

        "use_scheduler": use_scheduler,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "min_lr": min_lr,

        "use_target_standardization": use_target_standardization,

        "diagnostic_every": diagnostic_every,
        "save_val_predictions_every": save_val_predictions_every,
        "run_equivariance_check": run_equivariance_check,
        "equivariance_check_every": equivariance_check_every,
        "equivariance_check_batches": equivariance_check_batches,

        "run_eda": run_eda,
        "eda_num_graphs": eda_num_graphs,
        "eda_pairwise_distance_sample": eda_pairwise_distance_sample,

        "checkpoint_name": checkpoint_name,
        "log_name": log_name,
        "submission_name": submission_name,
        "diagnostic_name": diagnostic_name,
        "val_prediction_name": val_prediction_name,
        "eda_name": eda_name,
    }

