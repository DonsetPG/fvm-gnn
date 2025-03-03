import os
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from models.gnn import HeatEquationGNN

from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler


def _downsample(
    X: np.ndarray, y: np.ndarray, max_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_points:
        return X, y
    indices = np.random.choice(len(X), size=max_points, replace=False)
    return X[indices], y[indices]


def _run_pysr(X: np.ndarray, y: np.ndarray, names: Iterable[str], max_points: int):

    X_ds, y_ds = _downsample(X, y, max_points)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_ds)
    y_scaled = scaler_y.fit_transform(y_ds.reshape(-1, 1)).ravel()

    model = PySRRegressor(
        populations=32,
        niterations=30,
        binary_operators=["plus", "sub", "mult", "div"],
        unary_operators=["neg", "square"],
        procs=os.cpu_count() or 1,
        maxsize=20,
        progress=False,
    )

    model.fit(X_scaled, y_scaled, variable_names=list(names))
    return model


def _collect_mlp_io(
    model: HeatEquationGNN, loader: DataLoader, mlp_name: str, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    conv_layer = model.conv1
    mlp_module = getattr(conv_layer, mlp_name)

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    def _hook(_module, module_input, module_output):
        features.append(module_input[0].detach().cpu().numpy())
        targets.append(module_output.detach().cpu().numpy())

    handle = mlp_module.register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            model(batch.to(device))

    handle.remove()

    X = np.concatenate(features, axis=0)
    y = np.concatenate(targets, axis=0).ravel()
    return X, y


def run_symbolic_regression_for_model(
    model: HeatEquationGNN,
    loader: DataLoader,
    device: torch.device,
    *,
    max_points: int,
    output_dir: str,
    use_wandb: bool,
) -> Dict[str, str]:

    os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, str] = {}
    mlp_configs = {
        "mlp_msg": {
            "names": ["masked_flux"],
        },
        "mlp_upd": {
            "names": [
                "temperature",
                "is_boundary",
                "dt_source",
                "inv_volume",
                "masked_flux",
            ],
        },
    }

    for mlp_name, cfg in mlp_configs.items():
        X, y = _collect_mlp_io(model, loader, mlp_name, device)
        sr_model = _run_pysr(X, y, cfg["names"], max_points)
        equations = sr_model.equations_
        out_file = os.path.join(output_dir, f"{mlp_name}_equations.csv")
        equations.to_csv(out_file, index=False)

        best_eq = getattr(sr_model, "get_best", None)
        best_equation = ""
        if callable(best_eq):
            try:
                best_row = best_eq()
                if hasattr(best_row, "get"):
                    best_equation = str(best_row.get("equation", ""))
                else:
                    best_equation = str(getattr(best_row, "equation", ""))
            except Exception:
                best_equation = ""
        if not best_equation and not equations.empty:
            best_equation = str(equations.iloc[0].get("equation", ""))

        results[mlp_name] = best_equation

    return results
