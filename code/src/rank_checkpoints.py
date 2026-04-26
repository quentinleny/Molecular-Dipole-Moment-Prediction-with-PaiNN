# rank_checkpoints.py
# Rank checkpoint files by stored best validation MAE.

from pathlib import Path
import torch

import config


def get_checkpoint_mae(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    candidate_keys = [
        "best_val_mae",
        "val_mae",
        "best_mae",
        "mae",
    ]

    for key in candidate_keys:
        if key in ckpt:
            return float(ckpt[key])

    if "metrics" in ckpt:
        for key in candidate_keys:
            if key in ckpt["metrics"]:
                return float(ckpt["metrics"][key])

    if "config" in ckpt and "best_val_mae" in ckpt["config"]:
        return float(ckpt["config"]["best_val_mae"])

    return None


def main():
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("*.pt"))

    rows = []

    for path in checkpoint_paths:
        mae = get_checkpoint_mae(path)

        if mae is not None:
            rows.append((mae, path.name))
        else:
            rows.append((float("inf"), path.name))

    rows = sorted(rows, key=lambda x: x[0])

    print("\nTop 5 checkpoints by validation MAE:")
    print("-" * 80)

    for rank, (mae, name) in enumerate(rows[:5], start=1):
        mae_str = "NA" if mae == float("inf") else f"{mae:.6f}"
        print(f"{rank:>2}. {mae_str} | {name}")

    missing = [name for mae, name in rows if mae == float("inf")]

    if missing:
        print("\nCheckpoints missing readable MAE:")
        for name in missing:
            print(name)


if __name__ == "__main__":
    main()
