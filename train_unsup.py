import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from data.datasets import MinimalFVMDataset, OODFVMDataset
from losses import pde_residual_loss
from models.gnn import HeatEquationGNN
from symbolic_regression import (
    run_symbolic_regression_for_model,
)
import wandb


@dataclass
class Metrics:
    train_total: float
    train_residual: float
    train_bc: float
    val_mse: float
    test_mse: float
    ood_mse: float | None
    symbolic_regression: Dict[str, str] | None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def split_dataset(dataset, train_frac=0.7, val_frac=0.15):
    total_len = len(dataset)
    train_len = int(total_len * train_frac)
    val_len = int(total_len * val_frac)
    test_len = total_len - train_len - val_len
    return torch.utils.data.random_split(dataset, [train_len, val_len, test_len])


def get_boundary_mask(boundary_attr: torch.Tensor) -> torch.Tensor:
    mask = boundary_attr
    if mask.ndim > 1:
        mask = mask.view(-1)
    if mask.dtype != torch.bool:
        mask = mask >= 0.5
    return mask.bool()


def train_one_epoch(
    model: HeatEquationGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    dt: float,
    bc_weight: float,
    lambda_l1: float,
    sym_norm: bool,
    use_wandb: bool,
) -> Dict[str, float]:
    model.train()
    running = {"res": 0.0, "bc": 0.0, "total": 0.0}
    count_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        T_n = batch.x[:, 0:1]
        boundary_mask = get_boundary_mask(batch.boundary_mask)
        S_n = batch.x[:, 2:3] / dt
        edge_lengths = batch.edge_lengths
        edge_deltas = batch.edge_deltas

        T_pred = model(batch)

        res_mse, bc_mse, _ = pde_residual_loss(
            T_pred_raw=T_pred,
            T_n=T_n,
            S_n=S_n,
            edge_index=batch.edge_index,
            edge_lengths=edge_lengths,
            edge_deltas=edge_deltas,
            boundary_mask=boundary_mask,
            dt=dt,
            alpha=alpha,
            sym_norm=sym_norm,
        )

        l1_term = torch.tensor(0.0, device=T_pred.device)
        if lambda_l1 > 0:
            l1_term = model.l1_regularization() * lambda_l1

        loss = res_mse + bc_weight * bc_mse + l1_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_nodes = batch.num_nodes
        running["res"] += res_mse.item() * n_nodes
        running["bc"] += bc_mse.item() * n_nodes
        running["total"] += loss.item() * n_nodes
        count_nodes += n_nodes

        if use_wandb:
            wandb.log(
                {
                    "train_step/residual_mse": res_mse.item(),
                    "train_step/bc_mse": bc_mse.item(),
                    "train_step/total": loss.item(),
                    "train_step/l1": l1_term.item(),
                },
                commit=False,
            )

    for key in running:
        running[key] /= max(count_nodes, 1)
    return running


@torch.no_grad()
def evaluate_supervised(
    model: HeatEquationGNN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    criterion = torch.nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        total_loss += criterion(pred, batch.y).item()
        total_nodes += batch.num_nodes

    if total_nodes == 0:
        return float("nan")
    return total_loss / total_nodes


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
        DataLoader(ood_dataset, batch_size=1) if ood_dataset is not None else None
    )

    sample = dataset[0]
    model = HeatEquationGNN(
        in_channels=sample.x.size(-1),
        hidden_channels=args.hidden,
        edge_dim=sample.edge_attr.size(-1),
        dropout=args.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    use_wandb = args.wandb and wandb is not None
    if args.wandb and wandb is None:
        print("wandb requested but not available; continuing without logging.")
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.watch(model, log="all", log_freq=50)

    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            alpha=args.alpha,
            dt=args.dt,
            bc_weight=args.bc_weight,
            lambda_l1=args.l1,
            sym_norm=args.sym_norm,
            use_wandb=use_wandb,
        )
        val_mse = evaluate_supervised(model, val_loader, device)
        test_mse = evaluate_supervised(model, test_loader, device)
        metrics = {
            "train/residual_mse": train_metrics["res"],
            "train/bc_mse": train_metrics["bc"],
            "train/total": train_metrics["total"],
            "val/mse": val_mse,
            "test/mse": test_mse,
        }
        log_metrics(epoch, metrics, use_wandb)

        if val_mse < best_val:
            best_val = val_mse
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        val_mse = evaluate_supervised(model, val_loader, device)
        test_mse = evaluate_supervised(model, test_loader, device)
    else:
        val_mse = float("nan")
        test_mse = float("nan")

    ood_mse = None
    if ood_loader is not None:
        ood_mse = evaluate_supervised(model, ood_loader, device)

    symbolic_results: Dict[str, str] | None = None
    if args.symbolic_regression:
        sr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
        output_dir = os.path.join("symbolic_regression", args.run_name)
        symbolic_results = run_symbolic_regression_for_model(
            model,
            sr_loader,
            device,
            max_points=args.sr_max_points,
            output_dir=output_dir,
            use_wandb=use_wandb,
        )
        for mlp_name, equation in symbolic_results.items():
            print(json.dumps({"symbolic_regression": mlp_name, "equation": equation}))

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", f"{args.run_name}.pt")
    torch.save({"model": model.state_dict(), "config": vars(args)}, checkpoint_path)

    if use_wandb:
        wandb.save(checkpoint_path)
        wandb.summary["val/mse"] = val_mse
        wandb.summary["test/mse"] = test_mse
        if ood_mse is not None:
            wandb.summary["ood/mse"] = ood_mse
        if symbolic_results is not None:
            for mlp_name, equation in symbolic_results.items():
                wandb.summary[f"symbolic/{mlp_name}"] = equation

    return Metrics(
        train_total=train_metrics["total"],
        train_residual=train_metrics["res"],
        train_bc=train_metrics["bc"],
        val_mse=val_mse,
        test_mse=test_mse,
        ood_mse=ood_mse,
        symbolic_regression=symbolic_results,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train with PDE-residual supervision.")
    parser.add_argument(
        "--run_name", default="unsup", help="Checkpoint name and W&B run name."
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--bc_weight", type=float, default=10.0)
    parser.add_argument(
        "--sym_norm",
        action="store_true",
        help="Use the symmetric normalised Laplacian in the residual.",
    )
    parser.add_argument("--source_min", type=float, default=5.0)
    parser.add_argument("--source_max", type=float, default=15.0)
    parser.add_argument("--train_samples", type=int, default=150)
    parser.add_argument("--ood_samples", type=int, default=50)
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
        "--symbolic_regression",
        action="store_true",
        help="Run PySR on the trained model to recover closed-form equations.",
    )
    parser.add_argument(
        "--sr_max_points",
        type=int,
        default=5000,
        help="Maximum number of activations to feed into symbolic regression.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> Tuple[Metrics, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(args=list(argv) if argv is not None else None)
    metrics = train(args)
    return metrics, args


if __name__ == "__main__":
    main()
