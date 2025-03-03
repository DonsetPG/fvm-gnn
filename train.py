from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from data.datasets import MinimalFVMDataset, OODFVMDataset
from models.gnn import HeatEquationGNN
from symbolic_regression import (
    run_symbolic_regression_for_model,
    symbolic_regression_available,
)

import wandb


@dataclass
class Metrics:
    train_loss: float
    val_mse: float
    test_mse: float
    ood_mse: float | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, train_frac=0.7, val_frac=0.15):
    total_len = len(dataset)
    train_len = int(total_len * train_frac)
    val_len = int(total_len * val_frac)
    test_len = total_len - train_len - val_len
    return torch.utils.data.random_split(dataset, [train_len, val_len, test_len])


def evaluate(model: HeatEquationGNN, loader: DataLoader, device: torch.device) -> float:
    criterion = nn.MSELoss()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def log_metrics(step: int, metrics: Dict[str, float], use_wandb: bool) -> None:
    if use_wandb and wandb is not None:
        wandb.log(metrics, step=step)
    else:
        print(json.dumps({"step": step, **metrics}))


def train(args: argparse.Namespace) -> Metrics:
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    dataset = MinimalFVMDataset(
        num_samples=args.train_samples,
        alpha=args.alpha,
        dt=args.dt,
        source_value_min=args.source_min,
        source_value_max=args.source_max,
    )
    train_ds, val_ds, test_ds = split_dataset(dataset)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    ood_dataset = None
    if args.ood_samples > 0:
        ood_dataset = OODFVMDataset(
            num_samples=args.ood_samples,
            alpha=args.alpha,
            dt=args.dt,
            source_value_min=args.source_min,
            source_value_max=args.source_max,
            mesh_max_edge_size=args.mesh_max_edge,
        )
        if len(ood_dataset) == 0:
            ood_dataset = None
    ood_loader = (
        DataLoader(ood_dataset, batch_size=args.batch_size)
        if ood_dataset is not None
        else None
    )

    sample = dataset[0]
    model = HeatEquationGNN(
        in_channels=sample.x.size(-1),
        hidden_channels=args.hidden,
        edge_dim=sample.edge_attr.size(-1),
        dropout=args.dropout,
    ).to(device)

    summary_path = os.path.join("artifacts", f"{args.run_name}_model_summary.png")

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    use_wandb = args.wandb and wandb is not None
    if args.wandb and wandb is None:
        print("wandb requested but not available; continuing without logging.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )
        if os.path.exists(summary_path):
            wandb.log({"model/summary": wandb.Image(summary_path)}, step=0)

    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            if args.l1 > 0:
                loss = loss + args.l1 * model.l1_regularization()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(train_loader), 1)
        val_mse = evaluate(model, val_loader, device)
        test_mse = evaluate(model, test_loader, device)
        ood_mse = (
            evaluate(model, ood_loader, device) if ood_loader is not None else None
        )

        log_metrics(
            epoch,
            {
                "train/loss": epoch_loss,
                "val/mse": val_mse,
                "test/mse": test_mse,
                **({"ood/mse": ood_mse} if ood_mse is not None else {}),
            },
            use_wandb,
        )

        if val_mse < best_val:
            best_val = val_mse
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        val_mse = evaluate(model, val_loader, device)
        test_mse = evaluate(model, test_loader, device)
        ood_mse = (
            evaluate(model, ood_loader, device) if ood_loader is not None else None
        )

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", f"{args.run_name}.pt")
    torch.save({"model": model.state_dict(), "config": vars(args)}, checkpoint_path)

    if use_wandb:
        wandb.save(checkpoint_path)
        wandb.summary["test/mse"] = test_mse
        if ood_mse is not None:
            wandb.summary["ood/mse"] = ood_mse

    if args.symbolic_regression:
        sr_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        sr_results = run_symbolic_regression_for_model(
            model,
            sr_loader,
            device=device,
            max_points=args.sr_max_points,
            output_dir=args.sr_output_dir,
            use_wandb=use_wandb,
        )
        if use_wandb and sr_results:
            wandb.log(
                {f"symbolic/{name}": eq for name, eq in sr_results.items()},
                step=args.epochs,
            )

    return Metrics(
        train_loss=epoch_loss,
        val_mse=val_mse,
        test_mse=test_mse,
        ood_mse=ood_mse,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a 2-layer GNN for FVM supervision."
    )
    parser.add_argument(
        "--run_name", default="dense", help="Checkpoint name and W&B run name."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1)
    parser.add_argument("--source_min", type=float, default=5.0)
    parser.add_argument("--source_max", type=float, default=15.0)
    parser.add_argument("--train_samples", type=int, default=150)
    parser.add_argument("--ood_samples", type=int, default=20)
    parser.add_argument("--mesh_max_edge", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training even if CUDA is available.",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument("--wandb_project", default="heat-gnn")
    parser.add_argument(
        "--symbolic_regression", action="store_true", help="Run PySR on trained MLPs."
    )
    parser.add_argument(
        "--sr_max_points",
        type=int,
        default=20000,
        help="Maximum samples for symbolic regression.",
    )
    parser.add_argument(
        "--sr_output_dir",
        default="symbolic_regression",
        help="Directory for symbolic regression outputs.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> Tuple[Metrics, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(args=list(argv) if argv is not None else None)
    metrics = train(args)
    return metrics, args


if __name__ == "__main__":
    main()
