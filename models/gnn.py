import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class FVMConv(MessagePassing):

    def __init__(
        self, node_features: int, edge_features: int, hidden_channels: int
    ) -> None:
        super().__init__(aggr="add")

        msg_input_dim = edge_features
        self.mlp_msg = nn.Sequential(
            nn.Linear(msg_input_dim, 1),
        )

        upd_input_dim = node_features + 1
        self.mlp_upd = nn.Sequential(
            nn.Linear(upd_input_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return x[:, 0:1] + self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:

        del x_i, x_j
        return self.mlp_msg(edge_attr)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        scaled_aggr = aggr_out * x[:, 3].unsqueeze(-1)
        input_upd = torch.cat([x, scaled_aggr], dim=-1)
        return self.mlp_upd(input_upd)


class HeatEquationGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del dropout

        self.conv1 = FVMConv(in_channels, edge_dim, hidden_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        return self.conv1(x, edge_index, edge_attr)

    def l1_regularization(self) -> torch.Tensor:
        l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l1 = l1 + param.abs().sum()
        return l1
