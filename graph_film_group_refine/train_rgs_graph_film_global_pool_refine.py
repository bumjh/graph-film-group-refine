"""RG Graph-FiLM refinement with global graph pooling.

Ablation against ``train_rgs_graph_film_refine.py``. Everything is kept the
same except the clinical GCN output is pooled globally across all auxiliary
nodes instead of using superior/inferior/NVT group-aware pooling.

    RGB image -> RGB MTL baseline -> z_rgb, RGB feature F_rgb
    F_rgb -> task-specific aux latent h_i -> aux supervision logits
    h_i -> fixed clinical GCN -> global node pooling
    global context -> gamma/beta -> conditional FiLM on RG feature
    z_final = RG classifier(F_rg)

Default mode is unfrozen end-to-end fine-tuning from the RGB MTL checkpoint.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from . import train_rgs_conditional_refine as conditional
    from . import train_rgs_graph_film_refine as grouped
except ImportError:
    import train_rgs_conditional_refine as conditional
    import train_rgs_graph_film_refine as grouped


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "checkpoints" / "graph_film_global_pool_refine")


class GlobalPoolAuxLatentClinicalGCN(nn.Module):
    """Clinical GCN followed by one global graph context.

    The graph_dim/fusion_dim are intentionally identical to the group-aware
    version so this is a pooling ablation, not a capacity ablation.
    """

    def __init__(
        self,
        aux_task_names: Sequence[str],
        graph_dim: int,
        fusion_dim: int,
        dropout: float,
        gate_initial_scale: float,
    ) -> None:
        super().__init__()
        self.aux_task_names = [str(name) for name in aux_task_names]
        self.num_aux_tasks = len(self.aux_task_names)
        self.graph_dim = int(graph_dim)
        if self.num_aux_tasks < 1:
            raise ValueError("GlobalPoolAuxLatentClinicalGCN requires at least one auxiliary task.")

        adjacency = grouped.film.AuxGATReuseTaskFeatureMoEMTLHead._build_clinical_aux_adjacency(self.aux_task_names).float()
        self.register_buffer("gcn_adjacency", grouped.AuxLatentClinicalGCN._normalize_adjacency(adjacency), persistent=True)
        self.register_buffer("global_mask", torch.full((self.num_aux_tasks,), 1.0 / self.num_aux_tasks), persistent=False)

        self.task_embeddings = nn.Parameter(torch.zeros(self.num_aux_tasks, self.graph_dim))
        nn.init.normal_(self.task_embeddings, std=0.02)
        self.input_norm = nn.LayerNorm(self.graph_dim)
        self.gcn_transform = nn.Sequential(
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.graph_norm = nn.LayerNorm(self.graph_dim)
        self.context_pool = nn.Sequential(
            nn.LayerNorm(self.graph_dim),
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_film_head = nn.Sequential(
            nn.LayerNorm(self.graph_dim),
            nn.Linear(self.graph_dim, fusion_dim * 2),
        )
        scale = min(max(float(gate_initial_scale), 1e-4), 0.99)
        self.gate_scale_logit = nn.Parameter(torch.tensor(math.log(scale / (1.0 - scale))))

    def forward(self, aux_node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if aux_node_features.ndim != 3:
            raise ValueError(f"Expected aux node features [B,T,D], got {tuple(aux_node_features.shape)}")
        if aux_node_features.shape[1] != self.num_aux_tasks:
            raise ValueError(f"Expected {self.num_aux_tasks} aux node features, got {aux_node_features.shape[1]}")

        nodes = self.input_norm(aux_node_features + self.task_embeddings.unsqueeze(0))
        adjacency = self.gcn_adjacency.to(device=nodes.device, dtype=nodes.dtype)
        propagated = torch.einsum("ij,bjd->bid", adjacency, nodes)
        graph_nodes = self.graph_norm(nodes + self.gcn_transform(propagated))

        weights = self.global_mask.to(device=nodes.device, dtype=nodes.dtype)
        pooled = torch.einsum("t,btd->bd", weights, graph_nodes)
        graph_context = self.context_pool(pooled)
        gamma, beta = self.global_film_head(graph_context).chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)

        graph_matrix = adjacency.view(1, 1, adjacency.shape[0], adjacency.shape[1]).expand(nodes.size(0), 1, -1, -1)
        global_weight = torch.ones(1, device=nodes.device, dtype=nodes.dtype)
        return gamma, beta, graph_context, graph_matrix, gamma.unsqueeze(1), beta.unsqueeze(1), global_weight

    @property
    def gate_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scale_logit)


class GraphFiLMGlobalPoolRefineModel(grouped.GraphFiLMRefineModel):
    def __init__(
        self,
        args: argparse.Namespace,
        aux_tasks: int,
        aux_task_names: Sequence[str],
        geometry_dim: int,
    ) -> None:
        super().__init__(args=args, aux_tasks=aux_tasks, aux_task_names=aux_task_names, geometry_dim=geometry_dim)
        self.graph_gate = GlobalPoolAuxLatentClinicalGCN(
            aux_task_names=aux_task_names,
            graph_dim=args.moe_dim,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            gate_initial_scale=args.refine_initial_scale,
        )


def _cli_has_option(name: str) -> bool:
    import sys

    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = conditional.parse_args(argv)
    args.output_dir = DEFAULT_OUTPUT_DIR if not _cli_has_option("--output-dir") else args.output_dir
    if not _cli_has_option("--checkpoint"):
        args.checkpoint = str(Path(args.output_dir) / "best.pt")
    args.head_type = "graph_film_global_pool_refine"
    args.use_structural_prior = False
    args.prior_dropout = 0.0
    args.rgb_checkpoint = getattr(args, "rgb_checkpoint", None)
    args.freeze_rgb_baseline = bool(getattr(args, "freeze_rgb_baseline", False))
    return args


def main() -> None:
    args = parse_args()
    conditional.ConditionalRGRefineModel = GraphFiLMGlobalPoolRefineModel
    conditional.run(args)


if __name__ == "__main__":
    main()
