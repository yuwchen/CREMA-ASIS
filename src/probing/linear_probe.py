"""Linear probing: model, training loop, and per-layer experiment runner.

Consolidates the ``LinearProbe`` class, ``train_epoch``, ``evaluate``, and
``run_layer_probe_experiment`` functions that were duplicated across the
three feature-extraction scripts.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.probing.dataset import CachedEmbeddingDataset, collate_fn


# ---------------------------------------------------------------------------
# Linear probe model
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Single-layer linear classifier on top of frozen features."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Train / evaluate helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler=None,
    device: str = "cuda:0",
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of ``(avg_loss, accuracy)``.
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for features, labels in tqdm(dataloader, desc="Training", leave=False):
        features = features.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda:0",
) -> tuple[float, float, list, list]:
    """Evaluate the model.

    Returns:
        Tuple of ``(avg_loss, accuracy, predictions, labels)``.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for features, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        features = features.to(device).float()
        labels = labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(dataloader),
        accuracy_score(all_labels, all_preds),
        all_preds,
        all_labels,
    )


# ---------------------------------------------------------------------------
# Layer-sort key for mixed layer names (whisper, projector, int, MIMO)
# ---------------------------------------------------------------------------

def _layer_sort_key(x):
    """Sort layers: whisper → projector → numeric → MIMO / other strings."""
    if x == "whisper":
        return (0, 0, "")
    elif x == "projector":
        return (1, 0, "")
    elif isinstance(x, (int, float)):
        return (2, int(x), "")
    elif isinstance(x, str) and x.lstrip("-").isdigit():
        return (2, int(x), "")
    else:
        return (3, 0, str(x))


# ---------------------------------------------------------------------------
# Full layer-wise probing experiment
# ---------------------------------------------------------------------------

def run_layer_probe_experiment(
    task_name: str,
    train_embeddings: dict,
    val_embeddings: dict,
    test_embeddings: dict,
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    train_labels: list[str],
    val_labels: list[str],
    test_labels: list[str],
    label_to_idx: dict[str, int],
    pooling_strategies: list[str],
    learning_rates: list[float],
    batch_size: int = 64,
    num_epochs: int = 20,
    use_scheduler: bool = True,
    results_dir: str = "probe_results",
    lora_path: Optional[str] = None,
    evaluate_per_pooling: bool = True,
    device: str = "cuda:0",
) -> dict:
    """Run probing experiments across all layers found in the embeddings.

    For each layer, searches over *pooling_strategies* × *learning_rates*,
    selects the best configuration by validation accuracy, evaluates on
    the test set, and writes incremental CSV and TXT result files.

    Args:
        task_name: E.g. ``"Acoustic Emotion"`` or ``"Semantic Label"``.
        train_embeddings / val_embeddings / test_embeddings:
            ``{file_path: {layer_key: {"mean": tensor, ...}}}``.
        train_files / val_files / test_files:
            Ordered file lists (must match label order).
        train_labels / val_labels / test_labels:
            String labels (looked up in *label_to_idx*).
        label_to_idx: ``{label_string: int_index}``.
        pooling_strategies: E.g. ``["mean", "last"]``.
        learning_rates: E.g. ``[5e-2, 1e-2, 5e-3, 1e-3]``.
        batch_size: Probe training batch size.
        num_epochs: Probe training epochs per config.
        use_scheduler: Whether to use cosine-annealing LR scheduler.
        results_dir: Directory for result files.
        lora_path: If set, used in result filenames.
        evaluate_per_pooling: Whether to test-eval the best config
            for each pooling strategy individually.
        device: Torch device string.

    Returns:
        Dict ``{layer_key: {"best_config": ..., "test_acc": ..., ...}}``.
    """
    os.makedirs(results_dir, exist_ok=True)
    num_classes = len(label_to_idx)

    # Convert string labels to int indices
    train_idx = [label_to_idx[l] for l in train_labels]
    val_idx = [label_to_idx[l] for l in val_labels]
    test_idx = [label_to_idx[l] for l in test_labels]

    # Discover layers
    example = list(train_embeddings.values())[0]
    layer_keys = sorted(example.keys(), key=_layer_sort_key)

    # Detect feature dims per layer
    feature_dims = {}
    for lk in layer_keys:
        sample = example[lk]
        first_pool = next(iter(sample.values()))
        if isinstance(first_pool, torch.Tensor):
            feature_dims[lk] = first_pool.shape[0]
        else:  # numpy
            feature_dims[lk] = first_pool.shape[0]

    run_tag = os.path.basename(lora_path).split("-")[-1] if lora_path else "base"

    # Incremental TXT file
    txt_path = os.path.join(
        results_dir,
        f"{run_tag}_{task_name.lower().replace(' ', '_')}_incremental.txt",
    )
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write(f"{task_name} — Layer-by-Layer Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write("=" * 80 + "\n\n")

    print(f"\n{'=' * 80}")
    print(f"Probing: {task_name} | layers={layer_keys} | classes={num_classes}")
    print(f"{'=' * 80}")

    all_results: Dict = {}

    for layer_key in layer_keys:
        feat_dim = feature_dims[layer_key]
        print(f"\n--- Layer {layer_key}  (dim={feat_dim}) ---")

        # CSV for this layer's hyperparameter search
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        lk_str = str(layer_key) if isinstance(layer_key, int) else layer_key
        csv_path = os.path.join(
            results_dir,
            f"{run_tag}_layer{lk_str}_{task_name.lower().replace(' ', '_')}_{ts}.csv",
        )
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "Layer", "Pooling", "LR", "Val_Acc", "Test_Acc",
                "Val_Loss", "Test_Loss", "Best_Epoch", "Is_Best",
                "Train_Acc", "Train_Loss",
            ])

        best_val_acc = 0.0
        best_config = None
        best_per_pooling: Dict = {}

        for pooling in pooling_strategies:
            best_pool_val = 0.0
            best_pool_cfg = None

            for lr in learning_rates:
                # -- Build dataloaders --
                train_ds = CachedEmbeddingDataset(
                    train_embeddings, train_idx, layer_key, pooling, train_files
                )
                val_ds = CachedEmbeddingDataset(
                    val_embeddings, val_idx, layer_key, pooling, val_files
                )
                train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
                )
                val_loader = DataLoader(
                    val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
                )

                # -- Train --
                probe = LinearProbe(feat_dim, num_classes).to(device)
                optim = torch.optim.Adam(probe.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()
                sched = None
                if use_scheduler:
                    total_steps = len(train_loader) * num_epochs
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optim, T_max=total_steps, eta_min=lr * 0.01
                    )

                best_ep_val = 0.0
                best_ep_state = None
                best_ep_info: Dict = {}

                for epoch in range(num_epochs):
                    t_loss, t_acc = train_epoch(
                        probe, train_loader, optim, criterion, sched, device
                    )
                    v_loss, v_acc, _, _ = evaluate(
                        probe, val_loader, criterion, device
                    )
                    if v_acc > best_ep_val:
                        best_ep_val = v_acc
                        best_ep_state = probe.state_dict().copy()
                        best_ep_info = {
                            "epoch": epoch + 1,
                            "val_loss": v_loss,
                            "train_acc": t_acc,
                            "train_loss": t_loss,
                        }

                is_best = best_ep_val > best_val_acc
                if is_best:
                    best_val_acc = best_ep_val
                    best_config = {
                        "pooling": pooling, "lr": lr,
                        "model_state": best_ep_state,
                        **best_ep_info,
                    }

                if best_ep_val > best_pool_val:
                    best_pool_val = best_ep_val
                    best_pool_cfg = {
                        "pooling": pooling, "lr": lr,
                        "model_state": best_ep_state,
                        **best_ep_info,
                    }

                # Append CSV row
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        layer_key, pooling, lr, best_ep_val, "",
                        best_ep_info.get("val_loss", ""),
                        "", best_ep_info.get("epoch", ""), is_best,
                        best_ep_info.get("train_acc", ""),
                        best_ep_info.get("train_loss", ""),
                    ])

                del probe, optim, train_loader, val_loader

            # -- Per-pooling test eval --
            if evaluate_per_pooling and best_pool_cfg:
                test_ds = CachedEmbeddingDataset(
                    test_embeddings, test_idx, layer_key, pooling, test_files
                )
                test_loader = DataLoader(
                    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
                )
                eval_probe = LinearProbe(feat_dim, num_classes).to(device)
                eval_probe.load_state_dict(best_pool_cfg["model_state"])
                te_loss, te_acc, _, _ = evaluate(
                    eval_probe, test_loader, nn.CrossEntropyLoss(), device
                )
                best_per_pooling[pooling] = {
                    "config": best_pool_cfg,
                    "test_acc": te_acc, "test_loss": te_loss,
                }
                print(f"  {pooling} test acc: {te_acc:.4f}")
                del eval_probe, test_loader

        # -- Overall best → retrain on train+val, evaluate on test --
        if best_config is None:
            continue
        
        """
        full_ds = CachedEmbeddingDataset(
            {**train_embeddings, **val_embeddings},
            train_idx + val_idx,
            layer_key, best_config["pooling"],
            train_files + val_files,
        )
        """
        full_ds = CachedEmbeddingDataset(
            train_embeddings,
            train_idx,
            layer_key, best_config["pooling"],
            train_files,
        )
        
        test_ds = CachedEmbeddingDataset(
            test_embeddings, test_idx, layer_key, best_config["pooling"], test_files
        )
        full_loader = DataLoader(
            full_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        final_probe = LinearProbe(feat_dim, num_classes).to(device)
        optim = torch.optim.Adam(final_probe.parameters(), lr=best_config["lr"])
        criterion = nn.CrossEntropyLoss()
        sched = None
        if use_scheduler:
            total_steps = len(full_loader) * num_epochs
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=total_steps, eta_min=best_config["lr"] * 0.01
            )
        for _ in range(num_epochs):
            train_epoch(final_probe, full_loader, optim, criterion, sched, device)

        te_loss, te_acc, te_preds, te_labels = evaluate(
            final_probe, test_loader, criterion, device
        )
        print(f"  Layer {layer_key} FINAL test acc: {te_acc:.4f}")

        # Record
        all_results[layer_key] = {
            "best_config": best_config,
            "best_val_acc": best_val_acc,
            "test_acc": te_acc,
            "test_loss": te_loss,
            "best_per_pooling": best_per_pooling,
        }

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                f"{layer_key}_FINAL", best_config["pooling"], best_config["lr"],
                best_val_acc, te_acc, best_config.get("val_loss", ""),
                te_loss, "FINAL",
            ])

        # Incremental TXT
        with open(txt_path, "a") as f:
            f.write(f"Layer {layer_key}: pool={best_config['pooling']}, "
                    f"lr={best_config['lr']:.0e}, "
                    f"val={best_val_acc:.4f}, test={te_acc:.4f}\n")
            f.flush()

        del final_probe, full_loader, test_loader

    with open(txt_path, "a") as f:
        f.write(f"\nCompleted: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 80 + "\n")

    return all_results
