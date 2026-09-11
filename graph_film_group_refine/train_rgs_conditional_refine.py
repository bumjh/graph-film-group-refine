"""Conditional RG refinement.

Stage 1 is an RGB-only MTL baseline that produces a preliminary RG logit.
Stage 2 is activated softly for suspicious cases and adds an SSTC + auxiliary
clinical-GCN residual correction:

    p0 = sigmoid(z_rgb)
    g = sigmoid(k * (p0 - tau))
    z_final = z_rgb + g * lambda * delta_clinical

The correction branch uses the same selective clinical GCN prior as the TaFM
experiments, but final RG ranking remains anchored to the RGB baseline.
"""

import argparse
import copy
import importlib
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch
import torch.nn as nn

try:
    from . import train_justraigs_multitask_stagewise as base
    from . import train_rgs_film_stagewise as film
except ImportError:
    import train_justraigs_multitask_stagewise as base
    import train_rgs_film_stagewise as film

film = importlib.reload(film)

film.FINAL_REUSE_GROUP_SPECS = [
    ("Superior", ("DH", "BCLVS", "RNFLDS", "ANRS")),
    ("Inferior", ("DH", "BCLVI", "RNFLDI", "ANRI")),
    ("SxI interaction", ("DH", "BCLVS", "RNFLDS", "ANRS", "BCLVI", "RNFLDI", "ANRI")),
]
film.FINAL_REUSE_GROUP_SCALES = {
    "Superior": 1.0,
    "Inferior": 1.0,
    "SxI interaction": 0.5,
}


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "checkpoints" / "conditional_rg_refine")


def _copy_model_args(args: argparse.Namespace) -> argparse.Namespace:
    return copy.deepcopy(args)


class ConditionalRGRefineModel(nn.Module):
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
        self.correction_scale = nn.Parameter(torch.tensor(float(args.refine_initial_scale)).logit())
        self.use_channels_last = False

        shared_kwargs = dict(
            image_model_name=args.image_model_name,
            pretrained=bool(getattr(args, "pretrained", True)),
            aux_tasks=aux_tasks,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            aux_task_names=aux_task_names,
            geometry_dim=geometry_dim,
            geometry_hidden_dim=args.geometry_hidden_dim,
            num_experts=args.num_experts,
            moe_dim=args.moe_dim,
            moe_hidden_dim=args.moe_hidden_dim,
            task_tower_dim=args.task_tower_dim,
            prior_hidden_dim=args.prior_hidden_dim,
            prior_dropout=args.prior_dropout,
            prior_dilation_kernel=args.prior_dilation_kernel,
            prior_attention_dim=args.prior_attention_dim,
            prior_attention_heads=args.prior_attention_heads,
            prior_attention_stage=args.prior_attention_stage,
        )
        self.rgb_model = film.StructuralPriorModulationModel(
            **shared_kwargs,
            head_type="simple",
            structural_prior_mode=args.structural_prior_mode,
            use_structural_prior=False,
            structural_prior_integration="film",
            aux_delta_initial_scale=args.aux_delta_initial_scale,
        )
        self.correction_model = film.StructuralPriorModulationModel(
            **shared_kwargs,
            head_type="aux_gcn_reuse_mtl",
            structural_prior_mode="derived6",
            use_structural_prior=True,
            structural_prior_integration="spatial_task_channel_uncertainty_film",
            tafm_grouping=getattr(args, "tafm_grouping", "clinical"),
            aux_delta_initial_scale=args.aux_delta_initial_scale,
        )

    @property
    def sstc_uncertainty_modulator(self):
        return getattr(self.correction_model, "sstc_uncertainty_modulator", None)

    @property
    def task_aware_uncertainty_modulator(self):
        return getattr(self.correction_model, "task_aware_uncertainty_modulator", None)

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
        correction_outputs = self.correction_model(
            images,
            seg_input,
            seg_valid=seg_valid,
            geometry=geometry,
            geometry_mask=geometry_mask,
        )

        rgb_logits = rgb_outputs["final_logits"]
        preliminary_prob = torch.sigmoid(rgb_logits)
        gate_source = preliminary_prob.detach() if self.detach_refine_gate else preliminary_prob
        refine_gate = torch.sigmoid(self.refine_gate_sharpness * (gate_source - self.refine_threshold))

        clinical_delta = correction_outputs.get("aux_delta_logits")
        if clinical_delta is None:
            clinical_delta = correction_outputs["final_logits"] - correction_outputs.get(
                "final_base_logits",
                correction_outputs["final_logits"].detach(),
            )
        correction_scale = torch.sigmoid(self.correction_scale)
        final_logits = rgb_logits + refine_gate * correction_scale * clinical_delta

        outputs = dict(correction_outputs)
        outputs["final_logits"] = final_logits
        outputs["aux_logits"] = correction_outputs["aux_logits"]
        outputs["rgb_base_logits"] = rgb_logits
        outputs["preliminary_prob"] = preliminary_prob
        outputs["refine_gate"] = refine_gate
        outputs["conditional_clinical_delta"] = clinical_delta
        outputs["conditional_correction_scale"] = correction_scale.expand_as(rgb_logits)
        return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RGB baseline with conditional SSTC/clinical-GCN refinement.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--csv", type=str)
    parser.add_argument("--val-csv", type=str)
    parser.add_argument("--eval-csv", "--test-csv", dest="eval_csv", type=str)
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--seg-cache-dir", type=str)
    parser.add_argument("--split-manifest", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--allow-unlabeled-eval", action="store_true")
    parser.add_argument("--preds-csv", type=str)
    parser.add_argument("--metrics-csv", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--image-lr", type=float)
    parser.add_argument("--prior-lr", type=float)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--prior-dropout", type=float)
    parser.add_argument("--prior-dilation-kernel", type=int, default=45)
    parser.add_argument("--threshold-mode", choices=["sensitivity", "youden", "f1", "fixed"])
    parser.add_argument("--target-sensitivity", type=float)
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--gradcam-samples", type=int)
    parser.add_argument("--disable-save-gradcam", dest="save_gradcam", action="store_false")
    parser.add_argument("--balanced-final-sampler", action="store_true")
    parser.add_argument("--disable-balanced-aux-sampler", dest="balanced_aux_sampler", action="store_false")
    parser.add_argument("--disable-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--channels-last", dest="channels_last", action="store_true")
    parser.add_argument("--disable-channels-last", dest="channels_last", action="store_false")
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--build-seg-cache-only", action="store_true")
    parser.add_argument("--rebuild-seg-memmap", action="store_true")
    parser.add_argument("--refine-threshold", type=float, default=0.20)
    parser.add_argument("--refine-gate-sharpness", type=float, default=25.0)
    parser.add_argument("--refine-initial-scale", type=float, default=0.05)
    parser.add_argument("--disable-detach-refine-gate", dest="detach_refine_gate", action="store_false")
    parser.add_argument("--rgb-checkpoint", type=str, default=None)
    parser.add_argument("--freeze-rgb-baseline", action="store_true")
    parser.add_argument("--freeze-rgb-backbone", action="store_true")
    parser.add_argument("--rgb-lr-scale", type=float, default=1.0)

    overrides = vars(parser.parse_args(argv))
    args = film.parse_args([])
    args.output_dir = DEFAULT_OUTPUT_DIR
    args.checkpoint = str(Path(DEFAULT_OUTPUT_DIR) / "best.pt")
    args.stage = "stage3"
    args.aux_tasks = ",".join(base.ALL_AUX_COLUMNS)
    args.run_stage2_after_stage1 = False
    args.head_type = "conditional_rg_refine"
    args.flat_output_dir = True
    args.structural_prior_mode = "derived6"
    args.prior_attention_stage = 2
    args.prior_dilation_kernel = 45
    args.geometry_csv = None
    args.geometry_features = ""
    args.balanced_aux_sampler = True
    args.moe_dim = 128
    args.task_tower_dim = 64
    args.moe_balance_loss_weight = 0.01
    args.moe_entropy_loss_weight = 0.001
    args.aux_delta_initial_scale = 0.001
    args.refine_threshold = 0.20
    args.refine_gate_sharpness = 25.0
    args.refine_initial_scale = 0.05
    args.detach_refine_gate = True
    args.rgb_checkpoint = None
    args.freeze_rgb_baseline = False
    args.freeze_rgb_backbone = False
    args.rgb_lr_scale = 1.0

    for key, value in overrides.items():
        setattr(args, key, value)
    if "checkpoint" not in overrides:
        args.checkpoint = str(Path(args.output_dir) / "best.pt")
    return args


def _params_from_modules(modules: Iterable[nn.Module]) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for module in modules:
        params.extend(list(module.parameters()))
    return params


def build_optimizer(model: ConditionalRGRefineModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    rgb_params = list(model.rgb_model.parameters())
    if getattr(args, "freeze_rgb_baseline", False):
        for param in rgb_params:
            param.requires_grad = False
    elif getattr(args, "freeze_rgb_backbone", False):
        for param in model.rgb_model.image_encoder.parameters():
            param.requires_grad = False
    correction_model = getattr(model, "correction_model", None)
    correction_image_params = (
        []
        if correction_model is model.rgb_model
        else [param for param in correction_model.image_encoder.parameters() if param.requires_grad]
    )
    rgb_image_params = [
        param for param in model.rgb_model.image_encoder.parameters() if param.requires_grad
    ]
    prior_modules = [
        correction_model.prior_modulators if correction_model is not None else None,
        correction_model.shared_spatial_gate if correction_model is not None else None,
        correction_model.sstc_uncertainty_modulator if correction_model is not None else None,
        correction_model.task_channel_film if correction_model is not None else None,
    ]
    prior_params = _params_from_modules([module for module in prior_modules if module is not None])
    image_ids = {id(param) for param in [*correction_image_params, *rgb_image_params]}
    prior_ids = {id(param) for param in prior_params}
    head_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in image_ids and id(param) not in prior_ids
    ]
    groups = []
    if correction_image_params:
        groups.append({"params": correction_image_params, "lr": args.image_lr, "name": "image"})
    if rgb_image_params:
        groups.append({"params": rgb_image_params, "lr": args.image_lr * float(args.rgb_lr_scale), "name": "rgb_image"})
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr, "name": "head"})
    if prior_params:
        groups.append({"params": prior_params, "lr": args.prior_lr, "name": "prior"})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def load_rgb_checkpoint_into_baseline(
    checkpoint_path: Path,
    model: ConditionalRGRefineModel,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.rgb_model.state_dict()
    copied = 0
    skipped = 0
    filtered = {}
    for key, tensor in state.items():
        if key.startswith("rgb_model."):
            key = key[len("rgb_model."):]
        if key.startswith("model."):
            key = key[len("model."):]
        if key in target_state and target_state[key].shape == tensor.shape:
            filtered[key] = tensor
            copied += 1
        else:
            skipped += 1
    missing, unexpected = model.rgb_model.load_state_dict(filtered, strict=False)
    print(
        "Loaded RGB baseline checkpoint -> "
        f"path={checkpoint_path}, copied={copied}, skipped={skipped}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def run(args: argparse.Namespace) -> None:
    base.seed_everything(args.seed)
    selected_aux_tasks = base.resolve_selected_aux_tasks(args)
    base.configure_aux_tasks(selected_aux_tasks)
    args.geometry_feature_columns = [
        column.strip() for column in args.geometry_features.split(",") if column.strip()
    ] if args.geometry_csv else []
    args.output_dir = str(Path(args.output_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.build_seg_cache_only:
        train_df, val_df, test_df = base.build_train_val_test_dataframes(args)
        all_df = pd.concat([train_df, val_df, test_df], axis=0)
        cache = film.prepare_segmentation_memmap(args, all_df)
        print(f"Segmentation cache preparation complete -> entries={len(cache.lookup)}")
        return

    device = torch.device(args.device)
    amp_enabled = bool(device.type == "cuda" and not args.disable_amp)
    train_loader, val_loader, test_loader, final_pos_weight, aux_pos_weight, final_prior, aux_priors = film.build_film_loaders(args)
    model = ConditionalRGRefineModel(
        args,
        aux_tasks=len(base.AUX_COLUMNS),
        aux_task_names=base.AUX_COLUMNS,
        geometry_dim=len(args.geometry_feature_columns),
    ).to(device)
    model.use_channels_last = bool(args.channels_last and device.type == "cuda")
    if model.use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    film.apply_train_npmi_adjacency_if_available(model.correction_model, train_loader)
    if args.rgb_checkpoint:
        load_rgb_checkpoint_into_baseline(Path(args.rgb_checkpoint), model, device)
    if not args.disable_prior_bias_init:
        if not args.rgb_checkpoint:
            film.initialize_film_output_biases(model.rgb_model, final_prior=final_prior, aux_priors=aux_priors)
        if getattr(model, "correction_model", None) is not model.rgb_model:
            film.initialize_film_output_biases(model.correction_model, final_prior=final_prior, aux_priors=aux_priors)

    final_pos_weight = final_pos_weight.to(device)
    aux_pos_weight = aux_pos_weight.to(device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 1
    best_val_auroc = -float("inf")
    best_epoch = 0
    if args.resume:
        start_epoch, best_val_auroc = film.load_structural_prior_checkpoint(
            Path(args.resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        best_epoch = start_epoch - 1

    correction_labels = {
        "conditional_sstc_residual": "SSTC FiLM residual",
        "conditional_gcn_residual": "clinical GCN residual",
    }
    correction_label = correction_labels.get(getattr(args, "head_type", ""), "SSTC+clinical GCN")
    print(
        "Conditional RG refine setup -> "
        f"baseline=RGB MTL | correction={correction_label} | "
        f"refine_threshold={args.refine_threshold:g} | "
        f"gate_sharpness={args.refine_gate_sharpness:g} | "
        f"initial_scale={args.refine_initial_scale:g} | "
        f"rgb_checkpoint={args.rgb_checkpoint or 'none'} | "
        f"freeze_rgb={bool(args.freeze_rgb_baseline)} | "
        f"freeze_rgb_backbone={bool(getattr(args, 'freeze_rgb_backbone', False))} | "
        f"rgb_lr_scale={args.rgb_lr_scale:g} | "
        f"reuse_groups={film.FINAL_REUSE_GROUP_SPECS}"
    )

    metrics_history: List[Dict[str, float]] = []
    metrics_csv_path = Path(args.metrics_csv) if args.metrics_csv else output_dir / "metrics_history.csv"
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = film.train_one_epoch_film(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            amp_enabled,
            args.grad_clip_norm,
            final_pos_weight,
            aux_pos_weight,
            moe_balance_loss_weight=args.moe_balance_loss_weight,
            moe_entropy_loss_weight=args.moe_entropy_loss_weight,
        )
        val_metrics, _ = film.validate_film(
            model,
            val_loader,
            device,
            epoch,
            amp_enabled,
            final_pos_weight=final_pos_weight,
            aux_pos_weight=aux_pos_weight,
            graph_diagnostics_dir=output_dir / "graph_diagnostics",
            graph_split_name="val",
        )
        metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}
        lr_by_name = {
            str(group.get("name", f"group_{idx}")): float(group["lr"])
            for idx, group in enumerate(optimizer.param_groups)
        }
        if "image" in lr_by_name:
            metrics["lr_image"] = lr_by_name["image"]
        elif "rgb_image" in lr_by_name:
            metrics["lr_image"] = lr_by_name["rgb_image"]
        elif optimizer.param_groups:
            metrics["lr_image"] = float(optimizer.param_groups[0]["lr"])
        if "head" in lr_by_name:
            metrics["lr_head"] = lr_by_name["head"]
        elif optimizer.param_groups:
            metrics["lr_head"] = float(optimizer.param_groups[-1]["lr"])
        if "prior" in lr_by_name:
            metrics["lr_prior"] = lr_by_name["prior"]
        metrics_history.append(metrics.copy())
        base.save_metrics_history(metrics_history, metrics_csv_path)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | {base.format_metrics(metrics)}")

        current_auroc = float(val_metrics.get("val_auroc_final", float("nan")))
        if not math.isnan(current_auroc) and current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_epoch = epoch
            patience_counter = 0
            base.save_checkpoint(
                output_dir,
                "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=metrics,
                best_val_auroc=best_val_auroc,
                args=args,
            )
            print(f"New best checkpoint at epoch {epoch:03d} | val_auroc_final={best_val_auroc:.4f}")
        else:
            patience_counter += 1
            print(f"No validation AUROC improvement for {patience_counter} epoch(s). Best epoch={best_epoch:03d}")
        base.save_checkpoint(
            output_dir,
            "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            metrics=metrics,
            best_val_auroc=best_val_auroc,
            args=args,
        )
        scheduler.step()
        if args.early_stopping_patience and patience_counter >= args.early_stopping_patience:
            break

    best_path = output_dir / "best.pt"
    if best_path.exists():
        film.load_structural_prior_checkpoint(best_path, model=model, optimizer=None, scheduler=None, scaler=None, device=device)
        test_metrics, predictions = film.validate_film(
            model,
            test_loader,
            device,
            epoch=0,
            amp_enabled=amp_enabled,
            final_pos_weight=final_pos_weight,
            aux_pos_weight=aux_pos_weight,
            graph_diagnostics_dir=output_dir / "graph_diagnostics",
            graph_split_name="test",
        )
        print(f"Test checkpoint: {best_path}")
        print(base.format_metrics(base.rename_metric_prefix(test_metrics, "val_", "test_")))
        if predictions is not None:
            predictions.to_csv(output_dir / "test_predictions.csv", index=False)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
