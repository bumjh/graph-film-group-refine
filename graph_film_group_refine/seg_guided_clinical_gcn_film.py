"""Segmentation-guided Clinical GCN-FiLM experiment.

This wrapper reuses the RGGraphFiLM conditional graph-refinement code, but turns
on structural-prior FiLM before the auxiliary clinical GCN. The OD/OC segmenter
is used through the existing cached softmaps, not by running segmentation online
during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from . import train_rgs_graph_film_refine as graph_refine
except ImportError:
    import train_rgs_graph_film_refine as graph_refine


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "checkpoints" / "seg_guided_clinical_gcn_film")


_ORIGINAL_BUILD_STRUCTURAL_PRIOR = graph_refine.film.build_structural_prior
_ORIGINAL_STRUCTURAL_PRIOR_CHANNEL_COUNT = graph_refine.film.structural_prior_channel_count
_ORIGINAL_STRUCTURAL_PRIOR_CHANNEL_NAMES = graph_refine.film.structural_prior_channel_names
_ORIGINAL_STRUCTURAL_PRIOR_DESCRIPTION = graph_refine.film.structural_prior_description


def _build_anatomical5_prior(
    seg_input: torch.Tensor,
    dilation_kernel: int,
    include_morph_gradient: bool = False,
    rim_peri_only: bool = False,
) -> torch.Tensor:
    if include_morph_gradient or rim_peri_only:
        return _ORIGINAL_BUILD_STRUCTURAL_PRIOR(
            seg_input,
            dilation_kernel,
            include_morph_gradient=include_morph_gradient,
            rim_peri_only=rim_peri_only,
        )
    if seg_input.ndim != 4 or seg_input.shape[1] != 2:
        raise ValueError(f"Expected OD/OC softmaps with shape [B,2,H,W], got {tuple(seg_input.shape)}")
    if dilation_kernel < 1 or dilation_kernel % 2 == 0:
        raise ValueError("prior dilation kernel must be a positive odd integer")

    od_boundary_prob = seg_input[:, 0:1].clamp(0.0, 1.0)
    oc_prob = seg_input[:, 1:2].clamp(0.0, 1.0)
    od_prob = torch.maximum(graph_refine.film.fill_soft_ring_interior(od_boundary_prob), oc_prob)
    rim_prob = torch.relu(od_prob - oc_prob)
    dilated_disc = F.max_pool2d(
        od_prob,
        kernel_size=dilation_kernel,
        stride=1,
        padding=dilation_kernel // 2,
    )
    peripapillary_prob = torch.relu(dilated_disc - od_prob)

    _, _, height, _ = od_prob.shape
    y_coords = torch.arange(height, device=od_prob.device, dtype=od_prob.dtype).view(1, 1, height, 1)
    denominator = od_prob.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    centroid_y = (od_prob * y_coords).sum(dim=(2, 3), keepdim=True) / denominator
    superior_mask = (y_coords < centroid_y).to(dtype=od_prob.dtype)
    inferior_mask = 1.0 - superior_mask

    return torch.cat(
        [
            oc_prob,
            rim_prob * superior_mask,
            rim_prob * inferior_mask,
            peripapillary_prob * superior_mask,
            peripapillary_prob * inferior_mask,
        ],
        dim=1,
    )


def _anatomical5_channel_count(structural_prior_mode: str) -> int:
    if structural_prior_mode == "derived4":
        return 5
    return _ORIGINAL_STRUCTURAL_PRIOR_CHANNEL_COUNT(structural_prior_mode)


def _anatomical5_channel_names(structural_prior_mode: str):
    if structural_prior_mode == "derived4":
        return ["Cup", "Superior rim", "Inferior rim", "Superior peripapillary", "Inferior peripapillary"]
    return _ORIGINAL_STRUCTURAL_PRIOR_CHANNEL_NAMES(structural_prior_mode)


def _anatomical5_description(structural_prior_mode: str) -> str:
    if structural_prior_mode == "derived4":
        return "Cup+superior_rim+inferior_rim+superior_peripapillary+inferior_peripapillary(anatomical5)"
    return _ORIGINAL_STRUCTURAL_PRIOR_DESCRIPTION(structural_prior_mode)


def _enable_anatomical5_prior() -> None:
    graph_refine.film.build_structural_prior = _build_anatomical5_prior
    graph_refine.film.structural_prior_channel_count = _anatomical5_channel_count
    graph_refine.film.structural_prior_channel_names = _anatomical5_channel_names
    graph_refine.film.structural_prior_description = _anatomical5_description


class SegGuidedClinicalGCNFiLMModel(graph_refine.GraphFiLMRefineModel):
    """Graph-FiLM refinement with cached OD/OC structural-prior FiLM enabled."""

    def __init__(
        self,
        args,
        aux_tasks: int,
        aux_task_names: Sequence[str],
        geometry_dim: int,
    ) -> None:
        super().__init__(
            args=args,
            aux_tasks=aux_tasks,
            aux_task_names=aux_task_names,
            geometry_dim=geometry_dim,
        )

        film = graph_refine.film
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
            use_structural_prior=True,
            structural_prior_integration="film",
            prior_attention_dim=args.prior_attention_dim,
            prior_attention_heads=args.prior_attention_heads,
            prior_attention_stage=args.prior_attention_stage,
            aux_delta_initial_scale=args.aux_delta_initial_scale,
        )


def _cli_has_option(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def parse_args():
    args = graph_refine.conditional.parse_args()
    args.output_dir = DEFAULT_OUTPUT_DIR if not _cli_has_option("--output-dir") else args.output_dir
    if not _cli_has_option("--checkpoint"):
        args.checkpoint = str(Path(args.output_dir) / "best.pt")
    args.head_type = "seg_guided_clinical_gcn_film"
    args.use_structural_prior = True
    args.structural_prior_mode = "derived4"
    args.prior_dilation_kernel = 45
    args.prior_dropout = 0.0
    args.rgb_checkpoint = getattr(args, "rgb_checkpoint", None)
    args.freeze_rgb_baseline = True if not _cli_has_option("--freeze-rgb-baseline") else args.freeze_rgb_baseline
    return args


def main() -> None:
    args = parse_args()
    _enable_anatomical5_prior()
    graph_refine.conditional.ConditionalRGRefineModel = SegGuidedClinicalGCNFiLMModel
    graph_refine.conditional.run(args)


if __name__ == "__main__":
    main()
