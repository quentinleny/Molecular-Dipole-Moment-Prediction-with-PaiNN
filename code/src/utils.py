# utils.py
# Shared training, evaluation, prediction, and checkpoint utilities for HW4.

from pathlib import Path

import numpy as np
import torch
from torch_geometric import seed_everything


def set_seed(seed):
    """Set Python, NumPy, PyTorch, and PyG seeds."""
    seed_everything(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mae(pred, target):
    """Mean absolute error."""
    pred = pred.view(-1)
    target = target.view(-1)
    return torch.mean(torch.abs(pred - target))


def get_loss_fn(loss_name):
    """Return regression loss function."""
    if loss_name == "l1":
        return torch.nn.L1Loss()

    if loss_name == "smooth_l1":
        return torch.nn.SmoothL1Loss()

    if loss_name == "mse":
        return torch.nn.MSELoss()

    raise ValueError(f"Unsupported loss_name: {loss_name}")


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip_norm=None):
    """Train model for one epoch."""
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch).view(-1)
        target = batch.y.view(-1).float()

        loss = loss_fn(pred, target)
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        n = batch.num_graphs
        total_loss += loss.item() * n
        total_mae += mae(pred.detach(), target.detach()).item() * n
        total_examples += n

    avg_loss = total_loss / total_examples
    avg_mae = total_mae / total_examples

    return avg_loss, avg_mae


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model and return MAE, predictions, and targets."""
    model.eval()

    preds = []
    targets = []

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch).view(-1).detach().cpu()
        target = batch.y.view(-1).float().detach().cpu()

        preds.append(pred)
        targets.append(target)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    val_mae = mae(preds, targets).item()

    return val_mae, preds, targets


@torch.no_grad()
def predict(model, loader, device):
    """Predict test labels and return graph-name to prediction mapping."""
    model.eval()

    names = []
    preds = []

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch).view(-1).detach().cpu().numpy()

        batch_names = batch.name
        if isinstance(batch_names, str):
            batch_names = [batch_names]

        names.extend(list(batch_names))
        preds.extend(pred.tolist())

    if len(names) != len(preds):
        raise ValueError(f"Name/prediction length mismatch: {len(names)} vs {len(preds)}")

    return dict(zip(names, preds))


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    val_mae,
    config_dict=None,
    scheduler=None,
):
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_mae": val_mae,
    }

    if config_dict is not None:
        checkpoint["config"] = config_dict

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """Load model checkpoint. Optionally restore optimizer and scheduler."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def save_log(path, rows):
    """Save training log rows to CSV."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    return df


def prediction_summary(pred_map):
    """Return lightweight summary stats for prediction dictionary."""
    values = np.array(list(pred_map.values()), dtype=float)

    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }
