"""RG Graph-FiLM refinement.

This experiment keeps an RGB MTL baseline as the visual anchor, then derives
task-specific auxiliary latent features before auxiliary supervision. The latent
auxiliary nodes are passed through a fixed clinical GCN and converted into full
FiLM parameters for the RG feature:

    RGB image -> RGB MTL -> z_rgb, RGB feature F_rgb
    F_rgb -> task-specific aux latent h_i
    h_i -> aux supervision logits only
    h_i -> Clinical GCN -> superior/inferior/NVT group pooling
    group contexts -> group-aware gamma/beta -> summed FiLM parameters
    F_rg = F_rgb * (1 + conditional_gate * lambda * gamma) + conditional_gate * lambda * beta
    z_final = RG classifier(F_rg)

Compared with probability-driven graph gating, this version lets clinical
evidence reason over richer pre-head task features.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from . import train_justraigs_multitask_stagewise as base
    from . import train_rgs_film_stagewise as film
    from . import train_rgs_conditional_refine as conditional
except ImportError:
    import train_justraigs_multitask_stagewise as base
    import train_rgs_film_stagewise as film
    import train_rgs_conditional_refine as conditional

film = importlib.reload(film)


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "checkpoints" / "graph_film_refine")


class AuxLatentClinicalGCN(nn.Module):
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
            raise ValueError("AuxLatentClinicalGCN requires at least one auxiliary task.")

        adjacency = film.AuxGATReuseTaskFeatureMoEMTLHead._build_clinical_aux_adjacency(self.aux_task_names).float()
        self.register_buffer("gcn_adjacency", self._normalize_adjacency(adjacency), persistent=True)
        self.register_buffer("superior_mask", self._task_mask(["ANRS", "RNFLDS", "BCLVS", "DH"]), persistent=False)
        self.register_buffer("inferior_mask", self._task_mask(["ANRI", "RNFLDI", "BCLVI", "DH"]), persistent=False)
        self.register_buffer("nvt_mask", self._task_mask(["NVT"]), persistent=False)

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
            nn.LayerNorm(self.graph_dim * 3),
            nn.Linear(self.graph_dim * 3, self.graph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.group_film_heads = nn.ModuleDict(
            {
                "superior": nn.Sequential(nn.LayerNorm(self.graph_dim), nn.Linear(self.graph_dim, fusion_dim * 2)),
                "inferior": nn.Sequential(nn.LayerNorm(self.graph_dim), nn.Linear(self.graph_dim, fusion_dim * 2)),
                "nvt": nn.Sequential(nn.LayerNorm(self.graph_dim), nn.Linear(self.graph_dim, fusion_dim * 2)),
            }
        )
        self.context_film_head = nn.Sequential(
            nn.LayerNorm(self.graph_dim),
            nn.Linear(self.graph_dim, fusion_dim * 2),
        )
        self.group_mix_logits = nn.Parameter(torch.zeros(4))
        scale = min(max(float(gate_initial_scale), 1e-4), 0.99)
        self.gate_scale_logit = nn.Parameter(torch.tensor(math.log(scale / (1.0 - scale))))

    def _task_mask(self, names: Sequence[str]) -> torch.Tensor:
        selected = {str(name) for name in names}
        mask = torch.tensor([1.0 if name in selected else 0.0 for name in self.aux_task_names], dtype=torch.float32)
        if float(mask.sum()) <= 0.0:
            mask = torch.ones(self.num_aux_tasks, dtype=torch.float32)
        return mask / mask.sum().clamp_min(1.0)

    @staticmethod
    def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = torch.maximum(adjacency.float(), adjacency.float().transpose(0, 1))
        diagonal = torch.arange(adjacency.shape[0], device=adjacency.device)
        adjacency[diagonal, diagonal] = 1.0
        degree = adjacency.sum(dim=1).clamp_min(1e-6)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]

    def _pool_group(self, graph_nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(device=graph_nodes.device, dtype=graph_nodes.dtype)
        return torch.einsum("t,btd->bd", weights, graph_nodes)

    def forward(self, aux_node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if aux_node_features.ndim != 3:
            raise ValueError(f"Expected aux node features [B,T,D], got {tuple(aux_node_features.shape)}")
        if aux_node_features.shape[1] != self.num_aux_tasks:
            raise ValueError(f"Expected {self.num_aux_tasks} aux node features, got {aux_node_features.shape[1]}")
        nodes = self.input_norm(aux_node_features + self.task_embeddings.unsqueeze(0))
        adjacency = self.gcn_adjacency.to(device=nodes.device, dtype=nodes.dtype)
        propagated = torch.einsum("ij,bjd->bid", adjacency, nodes)
        graph_nodes = self.graph_norm(nodes + self.gcn_transform(propagated))

        superior = self._pool_group(graph_nodes, self.superior_mask)
        inferior = self._pool_group(graph_nodes, self.inferior_mask)
        nvt = self._pool_group(graph_nodes, self.nvt_mask)
        graph_context = self.context_pool(torch.cat([superior, inferior, nvt], dim=1))
        group_inputs = {
            "superior": superior,
            "inferior": inferior,
            "nvt": nvt,
            "global": graph_context,
        }
        group_pairs = []
        for group_name in ("superior", "inferior", "nvt"):
            group_pairs.append(self.group_film_heads[group_name](group_inputs[group_name]).chunk(2, dim=1))
        group_pairs.append(self.context_film_head(graph_context).chunk(2, dim=1))
        group_gamma = torch.stack([torch.tanh(pair[0]) for pair in group_pairs], dim=1)
        group_beta = torch.stack([torch.tanh(pair[1]) for pair in group_pairs], dim=1)
        group_weights = torch.softmax(self.group_mix_logits, dim=0).to(device=nodes.device, dtype=nodes.dtype)
        gamma = torch.einsum("g,bgf->bf", group_weights, group_gamma)
        beta = torch.einsum("g,bgf->bf", group_weights, group_beta)
        graph_matrix = adjacency.view(1, 1, adjacency.shape[0], adjacency.shape[1]).expand(nodes.size(0), 1, -1, -1)
        return gamma, beta, graph_context, graph_matrix, group_gamma, group_beta, group_weights

    @property
    def gate_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scale_logit)


class GraphFiLMRefineModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        aux_tasks: int,
        aux_task_names: Sequence[str],
        geometry_dim: int,
    ) -> None:
        super().__init__()
        self.refine_threshold = float(args.refine_threshold)
        self.refine_gate_sharpness = float(args.refine_gate_sharpness)
        self.detach_refine_gate = bool(args.detach_refine_gate)
        self.use_channels_last = False

        self.rgb_model = film.StructuralPriorModulationModel(
            image_model_name=args.image_model_name,
            pretrained=bool(getattr(args, "pretrained", True)),
            aux_tasks=aux_tasks,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            aux_task_names=aux_task_names,
            geometry_dim=geometry_dim,
            geometry_hidden_dim=args.geometry_hidden_dim,
            head_type="simple",
            num_experts=args.num_experts,
            moe_dim=args.moe_dim,
            moe_hidden_dim=args.moe_hidden_dim,
            task_tower_dim=args.task_tower_dim,
            prior_hidden_dim=args.prior_hidden_dim,
            prior_dropout=args.prior_dropout,
            prior_dilation_kernel=args.prior_dilation_kernel,
            structural_prior_mode=args.structural_prior_mode,
            use_structural_prior=False,
            structural_prior_integration="film",
            prior_attention_dim=args.prior_attention_dim,
            prior_attention_heads=args.prior_attention_heads,
            prior_attention_stage=args.prior_attention_stage,
            aux_delta_initial_scale=args.aux_delta_initial_scale,
        )
        self.graph_gate = AuxLatentClinicalGCN(
            aux_task_names=aux_task_names,
            graph_dim=args.moe_dim,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            gate_initial_scale=args.refine_initial_scale,
        )
        self.aux_feature_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(args.fusion_dim),
                    nn.Linear(args.fusion_dim, args.moe_dim),
                    nn.GELU(),
                    nn.Dropout(args.dropout),
                )
                for _ in range(aux_tasks)
            ]
        )
        self.aux_supervision_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(args.moe_dim),
                    nn.Linear(args.moe_dim, 1),
                )
                for _ in range(aux_tasks)
            ]
        )
        self.rg_classifier = nn.Sequential(
            nn.LayerNorm(args.fusion_dim),
            nn.Dropout(args.dropout),
            nn.Linear(args.fusion_dim, 1),
        )

    @property
    def sstc_uncertainty_modulator(self):
        return None

    @property
    def task_aware_uncertainty_modulator(self):
        return None

    @property
    def correction_model(self):
        return self.rgb_model

    def forward(
        self,
        images: torch.Tensor,
        seg_input: torch.Tensor,
        seg_valid: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
        geometry_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        rgb_outputs = self.rgb_model(
            images,
            seg_input,
            seg_valid=seg_valid,
            geometry=geometry,
            geometry_mask=geometry_mask,
        )
        rgb_logits = rgb_outputs["final_logits"]
        rgb_features = rgb_outputs["head_features"]

        aux_node_features = torch.stack([adapter(rgb_features) for adapter in self.aux_feature_adapters], dim=1)
        aux_logits = torch.stack(
            [head(aux_node_features[:, index]).squeeze(1) for index, head in enumerate(self.aux_supervision_heads)],
            dim=1,
        )

        preliminary_prob = torch.sigmoid(rgb_logits)
        gate_source = preliminary_prob.detach() if self.detach_refine_gate else preliminary_prob
        refine_gate = torch.sigmoid(self.refine_gate_sharpness * (gate_source - self.refine_threshold)).unsqueeze(1)

        gamma, beta, graph_context, graph_matrix, group_gamma, group_beta, group_weights = self.graph_gate(aux_node_features)
        scale = self.graph_gate.gate_scale
        alpha = refine_gate * scale
        modulated_features = rgb_features * (1.0 + alpha * gamma) + alpha * beta
        final_logits = self.rg_classifier(modulated_features).squeeze(1)

        outputs = dict(rgb_outputs)
        outputs["final_logits"] = final_logits
        outputs["aux_logits"] = aux_logits
        outputs["rgb_base_logits"] = rgb_logits
        outputs["preliminary_prob"] = preliminary_prob
        outputs["refine_gate"] = refine_gate.squeeze(1)
        outputs["graph_film_gamma_mean"] = gamma.mean(dim=1)
        outputs["graph_film_beta_mean"] = beta.mean(dim=1)
        outputs["graph_film_gate_scale"] = scale.expand_as(rgb_logits)
        outputs["graph_film_group_gamma_mean"] = group_gamma.mean(dim=2)
        outputs["graph_film_group_beta_mean"] = group_beta.mean(dim=2)
        outputs["graph_film_group_weights"] = group_weights.unsqueeze(0).expand(rgb_features.size(0), -1)
        outputs["graph_context"] = graph_context
        outputs["aux_node_features"] = aux_node_features
        outputs["aux_graph_attention"] = graph_matrix
        outputs["aux_graph_fixed"] = True
        outputs["head_features"] = modulated_features
        outputs["rgb_head_features"] = rgb_features
        return outputs


def _cli_has_option(name: str) -> bool:
    import sys

    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = conditional.parse_args(argv)
    args.output_dir = DEFAULT_OUTPUT_DIR if not _cli_has_option("--output-dir") else args.output_dir
    if not _cli_has_option("--checkpoint"):
        args.checkpoint = str(Path(args.output_dir) / "best.pt")
    args.head_type = "graph_film_refine"
    args.use_structural_prior = False
    args.prior_dropout = 0.0
    args.rgb_checkpoint = getattr(args, "rgb_checkpoint", None)
    args.freeze_rgb_baseline = True if not _cli_has_option("--freeze-rgb-baseline") else args.freeze_rgb_baseline
    return args


def main() -> None:
    args = parse_args()
    conditional.ConditionalRGRefineModel = GraphFiLMRefineModel
    conditional.run(args)


if __name__ == "__main__":
    main()
