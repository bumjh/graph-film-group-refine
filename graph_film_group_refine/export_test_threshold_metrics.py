import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

try:
    from . import train_justraigs_multitask_stagewise as base
    from . import train_rgs_film_stagewise as film
    from . import train_rgs_conditional_refine as conditional
    from . import train_rgs_graph_film_refine as graph_film
    from . import train_rgs_graph_film_global_pool_refine as graph_film_global
except ImportError:
    import train_justraigs_multitask_stagewise as base
    import train_rgs_film_stagewise as film
    import train_rgs_conditional_refine as conditional
    import train_rgs_graph_film_refine as graph_film
    import train_rgs_graph_film_global_pool_refine as graph_film_global


class QuietProgress:
    def __init__(self, iterable, **_kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **_kwargs) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final test metrics at Youden and specificity operating points.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to evaluate. Defaults to latest best.pt under --search-dir.")
    parser.add_argument("--search-dir", type=str, default=str(Path("checkpoints") / "graph_film_refine"))
    parser.add_argument("--output-xlsx", type=str, default=None)
    parser.add_argument("--target-specificity", type=float, default=0.95)
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=None, help="Override checkpoint batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--show-progress", dest="show_progress", action="store_true")
    parser.add_argument("--disable-progress", dest="show_progress", action="store_false")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--threshold-source",
        choices=["validation", "test"],
        default="validation",
        help=(
            "validation fixes operating thresholds before test bootstrap; "
            "test reproduces challenge-style test ROC operating points."
        ),
    )
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def latest_checkpoint(search_dir: Path) -> Path:
    candidates = sorted(search_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No best.pt found under {search_dir}")
    return candidates[0]


def film_default_args() -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["train_rgs_film_stagewise.py"]
        return film.parse_args()
    finally:
        sys.argv = old_argv


def load_run_args(checkpoint_path: Path, cli_args: argparse.Namespace) -> argparse.Namespace:
    args = film_default_args()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    if isinstance(checkpoint_args, argparse.Namespace):
        checkpoint_args = vars(checkpoint_args)
    for key, value in dict(checkpoint_args).items():
        setattr(args, key, value)

    args.checkpoint = str(checkpoint_path)
    args.eval_only = False
    args.resume = None
    args.warm_start_checkpoint = None
    args.save_gradcam = False
    args.num_workers = int(cli_args.num_workers)
    if cli_args.batch_size is not None:
        args.batch_size = int(cli_args.batch_size)
    if cli_args.device is not None:
        args.device = cli_args.device
    if cli_args.disable_amp:
        args.disable_amp = True

    checkpoint_aux_columns = checkpoint.get("aux_columns")
    if checkpoint_aux_columns:
        args.aux_tasks = ",".join(checkpoint_aux_columns)
    base.configure_aux_tasks(base.resolve_selected_aux_tasks(args))
    args.geometry_feature_columns = [
        column.strip() for column in str(args.geometry_features).split(",") if column.strip()
    ] if args.geometry_csv else []
    args.geometry_norm_stats = {}
    args.refine_threshold = float(getattr(args, "refine_threshold", 0.20))
    args.refine_gate_sharpness = float(getattr(args, "refine_gate_sharpness", 25.0))
    args.refine_initial_scale = float(getattr(args, "refine_initial_scale", 0.05))
    args.detach_refine_gate = bool(getattr(args, "detach_refine_gate", True))
    args.rgb_checkpoint = getattr(args, "rgb_checkpoint", None)
    args.freeze_rgb_baseline = bool(getattr(args, "freeze_rgb_baseline", False))
    args.rgb_lr_scale = float(getattr(args, "rgb_lr_scale", 1.0))
    return args


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    graph_film_model_classes = {
        "graph_film_refine": graph_film.GraphFiLMRefineModel,
        "graph_film_global_pool_refine": graph_film_global.GraphFiLMGlobalPoolRefineModel,
    }
    if getattr(args, "head_type", "") in graph_film_model_classes:
        model = graph_film_model_classes[getattr(args, "head_type")](
            args,
            aux_tasks=len(base.AUX_COLUMNS),
            aux_task_names=list(base.AUX_COLUMNS),
            geometry_dim=len(args.geometry_feature_columns),
        ).to(device)
        film.load_structural_prior_checkpoint(
            Path(args.checkpoint),
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            device=device,
        )
        return model
    if getattr(args, "head_type", "") == "conditional_rg_refine":
        model = conditional.ConditionalRGRefineModel(
            args,
            aux_tasks=len(base.AUX_COLUMNS),
            aux_task_names=list(base.AUX_COLUMNS),
            geometry_dim=len(args.geometry_feature_columns),
        ).to(device)
        film.load_structural_prior_checkpoint(
            Path(args.checkpoint),
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            device=device,
        )
        return model

    model = film.StructuralPriorModulationModel(
        image_model_name=args.image_model_name,
        pretrained=False,
        aux_tasks=len(base.AUX_COLUMNS),
        aux_task_names=list(base.AUX_COLUMNS),
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        geometry_dim=len(args.geometry_feature_columns),
        geometry_hidden_dim=args.geometry_hidden_dim,
        head_type=args.head_type,
        num_experts=args.num_experts,
        moe_dim=args.moe_dim,
        moe_hidden_dim=args.moe_hidden_dim,
        task_tower_dim=args.task_tower_dim,
        prior_hidden_dim=args.prior_hidden_dim,
        prior_dropout=args.prior_dropout,
        prior_dilation_kernel=args.prior_dilation_kernel,
        structural_prior_mode=getattr(args, "structural_prior_mode", "derived4"),
        use_structural_prior=getattr(args, "use_structural_prior", True),
        structural_prior_integration=getattr(args, "structural_prior_integration", "film"),
        tafm_grouping=getattr(args, "tafm_grouping", "clinical"),
        prior_attention_dim=getattr(args, "prior_attention_dim", 192),
        prior_attention_heads=getattr(args, "prior_attention_heads", 4),
        prior_attention_stage=getattr(args, "prior_attention_stage", 3),
    ).to(device)
    film.load_structural_prior_checkpoint(Path(args.checkpoint), model=model, optimizer=None, scheduler=None, scaler=None, device=device)
    return model


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(1.0, 0.0))
    return threshold


def specificity_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_specificity: float) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    valid = np.where(specificity >= target_specificity)[0]
    if len(valid) == 0:
        idx = int(np.argmax(specificity))
    else:
        best_tpr = np.max(tpr[valid])
        tied = valid[np.where(tpr[valid] == best_tpr)[0]]
        finite = tied[np.isfinite(thresholds[tied])]
        idx = int(finite[-1] if len(finite) else tied[-1])
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(1.0, 0.0))
    return threshold


def sensitivity_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_sensitivity: float) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    valid = np.where(tpr >= target_sensitivity)[0]
    if len(valid) == 0:
        idx = int(np.argmax(tpr))
    else:
        best_specificity = np.max(specificity[valid])
        tied = valid[np.where(specificity[valid] == best_specificity)[0]]
        finite = tied[np.isfinite(thresholds[tied])]
        idx = int(finite[0] if len(finite) else tied[0])
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(1.0, 0.0))
    return threshold


def f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(-y_prob, kind="mergesort")
    sorted_prob = y_prob[order]
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    total_positive = max(1, int(np.sum(sorted_true == 1)))
    distinct_end = np.r_[sorted_prob[:-1] != sorted_prob[1:], True]
    precision = tp / np.maximum(tp + fp, 1)
    sensitivity = tp / total_positive
    f1 = 2.0 * precision * sensitivity / np.maximum(precision + sensitivity, 1e-12)
    candidate_indices = np.flatnonzero(distinct_end)
    best_index = candidate_indices[int(np.argmax(f1[candidate_indices]))]
    return float(sorted_prob[best_index])


def summarize_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label: str,
    auroc: float | None = None,
    auprc: float | None = None,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2.0 * precision * sensitivity / max(1e-12, precision + sensitivity)
    return {
        "mode": label,
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "f1_score": float(f1),
        "accuracy": float((tp + tn) / max(1, len(y_true))),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "auroc": float(roc_auc_score(y_true, y_prob) if auroc is None else auroc),
        "auprc": float(average_precision_score(y_true, y_prob) if auprc is None else auprc),
        "pred_pos_rate": float(np.mean(y_pred)),
    }


def evaluate_threshold_modes(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float,
    target_sensitivity: float,
) -> pd.DataFrame:
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr

    youden_index = int(np.argmax(tpr - fpr))
    youden = float(roc_thresholds[youden_index])

    spec_valid = np.where(specificity >= target_specificity)[0]
    spec_candidates = spec_valid[np.where(tpr[spec_valid] == np.max(tpr[spec_valid]))[0]]
    spec_finite = spec_candidates[np.isfinite(roc_thresholds[spec_candidates])]
    spec_index = int(spec_finite[-1] if len(spec_finite) else spec_candidates[-1])
    spec_threshold = float(roc_thresholds[spec_index])

    sens_valid = np.where(tpr >= target_sensitivity)[0]
    sens_candidates = sens_valid[
        np.where(specificity[sens_valid] == np.max(specificity[sens_valid]))[0]
    ]
    sens_finite = sens_candidates[np.isfinite(roc_thresholds[sens_candidates])]
    sens_index = int(sens_finite[0] if len(sens_finite) else sens_candidates[0])
    sens_threshold = float(roc_thresholds[sens_index])

    thresholds = [
        ("youden", youden),
        (
            f"specificity_{target_specificity:.2f}",
            spec_threshold,
        ),
        (
            f"sensitivity_{target_sensitivity:.2f}",
            sens_threshold,
        ),
        ("f1_optimal", f1_threshold(y_true, y_prob)),
    ]
    return pd.DataFrame([
        summarize_at_threshold(y_true, y_prob, threshold, label, auroc=auroc, auprc=auprc)
        for label, threshold in thresholds
    ])


def evaluate_fixed_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    return pd.DataFrame([
        summarize_at_threshold(
            y_true,
            y_prob,
            threshold=float(threshold),
            label=label,
            auroc=auroc,
            auprc=auprc,
        )
        for label, threshold in thresholds.items()
    ])


def auxiliary_f1_thresholds(predictions: pd.DataFrame) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    rg_predictions = predictions[predictions["final_target"].astype(float) >= 0.5]
    for task in base.AUX_COLUMNS:
        target_column = f"{task}_target"
        probability_column = f"{task}_prob"
        mask_column = f"{task}_mask"
        if not {target_column, probability_column, mask_column}.issubset(rg_predictions.columns):
            continue
        valid = (
            rg_predictions[target_column].notna()
            & rg_predictions[probability_column].notna()
            & (rg_predictions[mask_column].astype(float) > 0)
        )
        task_frame = rg_predictions.loc[valid]
        if task_frame.empty:
            thresholds[task] = 0.5
            continue
        y_true = task_frame[target_column].astype(int).to_numpy()
        y_prob = task_frame[probability_column].astype(float).to_numpy()
        thresholds[task] = f1_threshold(y_true, y_prob) if np.unique(y_true).size == 2 else 0.5
    return thresholds


def evaluate_auxiliary_features(
    predictions: pd.DataFrame,
    thresholds: Dict[str, float],
    threshold_source: str,
) -> pd.DataFrame:
    rows = []
    rg_predictions = predictions[predictions["final_target"].astype(float) >= 0.5]
    for task in base.AUX_COLUMNS:
        target_column = f"{task}_target"
        probability_column = f"{task}_prob"
        mask_column = f"{task}_mask"
        if not {target_column, probability_column, mask_column}.issubset(rg_predictions.columns):
            continue
        valid = (
            rg_predictions[target_column].notna()
            & rg_predictions[probability_column].notna()
            & (rg_predictions[mask_column].astype(float) > 0)
        )
        task_frame = rg_predictions.loc[valid]
        if task_frame.empty:
            continue

        y_true = task_frame[target_column].astype(int).to_numpy()
        y_prob = task_frame[probability_column].astype(float).to_numpy()
        threshold = float(thresholds.get(task, 0.5))
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        rows.append({
            "task": task,
            "scope": "ground_truth_RG_and_valid_agreement_mask",
            "threshold_source": threshold_source,
            "threshold": threshold,
            "valid_n": int(len(y_true)),
            "positive_n": int(np.sum(y_true == 1)),
            "negative_n": int(np.sum(y_true == 0)),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float((tp + tn) / max(1, len(y_true))),
            "specificity": float(tn / max(1, tn + fp)),
            "auroc": (
                float(roc_auc_score(y_true, y_prob))
                if np.unique(y_true).size == 2
                else float("nan")
            ),
            "auprc": (
                float(average_precision_score(y_true, y_prob))
                if np.sum(y_true == 1) > 0
                else float("nan")
            ),
            "pred_pos_rate": float(np.mean(y_pred)),
        })
    return pd.DataFrame(rows)


def patient_cluster_bootstrap(
    eval_df: pd.DataFrame,
    target_specificity: float,
    target_sensitivity: float,
    replicates: int,
    confidence_level: float,
    seed: int,
    show_progress: bool,
    fixed_thresholds: Dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if replicates <= 0:
        return pd.DataFrame(), pd.DataFrame()
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("--confidence-level must be between 0 and 1.")
    if "patient_group" not in eval_df:
        raise ValueError("patient_group is required for patient-cluster bootstrap.")

    group_indices = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in eval_df.groupby("patient_group", sort=False)
    ]
    rng = np.random.default_rng(seed)
    records = []
    iterator = range(replicates)
    iterator = tqdm(iterator, desc="Patient bootstrap CI", disable=not show_progress)
    for replicate in iterator:
        sampled_groups = rng.integers(0, len(group_indices), size=len(group_indices))
        sampled_indices = np.concatenate([group_indices[index] for index in sampled_groups])
        sample = eval_df.iloc[sampled_indices]
        y_true = sample["final_target"].astype(int).to_numpy()
        if np.unique(y_true).size < 2:
            continue
        y_prob = sample["final_prob"].astype(float).to_numpy()
        if fixed_thresholds is None:
            replicate_summary = evaluate_threshold_modes(
                y_true,
                y_prob,
                target_specificity=target_specificity,
                target_sensitivity=target_sensitivity,
            )
        else:
            replicate_summary = evaluate_fixed_thresholds(y_true, y_prob, fixed_thresholds)
        replicate_summary.insert(0, "replicate", replicate)
        records.append(replicate_summary)

    if not records:
        raise RuntimeError("No valid bootstrap replicates contained both outcome classes.")
    distribution = pd.concat(records, ignore_index=True)
    alpha = 1.0 - confidence_level
    metric_columns = [
        "threshold",
        "f1_score",
        "accuracy",
        "specificity",
        "sensitivity",
        "auroc",
        "auprc",
        "pred_pos_rate",
    ]
    ci_rows = []
    for mode, group in distribution.groupby("mode", sort=False):
        row: Dict[str, float] = {
            "mode": mode,
            "confidence_level": confidence_level,
            "valid_replicates": int(group["replicate"].nunique()),
        }
        for metric in metric_columns:
            row[f"{metric}_ci_lower"] = float(group[metric].quantile(alpha / 2.0))
            row[f"{metric}_ci_upper"] = float(group[metric].quantile(1.0 - alpha / 2.0))
        ci_rows.append(row)
    return pd.DataFrame(ci_rows), distribution


def main() -> None:
    cli_args = parse_args()
    checkpoint_path = Path(cli_args.checkpoint) if cli_args.checkpoint else latest_checkpoint(Path(cli_args.search_dir))
    checkpoint_path = checkpoint_path.resolve()
    output_xlsx = Path(cli_args.output_xlsx) if cli_args.output_xlsx else checkpoint_path.parent / "test_threshold_metrics.xlsx"

    args = load_run_args(checkpoint_path, cli_args)
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    if not cli_args.show_progress:
        film.tqdm = QuietProgress

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Building test loader from checkpoint args | stage={args.stage} | head={args.head_type}")
    _, val_loader, test_loader, final_pos_weight, aux_pos_weight, _, _ = film.build_film_loaders(args)
    print(f"Test loader ready -> samples={len(test_loader.dataset)}, batches={len(test_loader)}")
    print("Loading checkpoint model...")
    model = build_model(args, device)
    print(f"Evaluating test set -> device={device}, amp={amp_enabled}")

    final_pos_weight = final_pos_weight.to(device)
    aux_pos_weight = aux_pos_weight.to(device)
    val_predictions = None
    if cli_args.threshold_source == "validation":
        print("Evaluating validation set to fix operating thresholds...")
        _, val_predictions = film.validate_film(
            model,
            val_loader,
            device,
            epoch=0,
            amp_enabled=amp_enabled,
            final_pos_weight=final_pos_weight,
            aux_pos_weight=aux_pos_weight,
            final_threshold=0.5,
        )
    metrics, predictions = film.validate_film(
        model,
        test_loader,
        device,
        epoch=0,
        amp_enabled=amp_enabled,
        final_pos_weight=final_pos_weight,
        aux_pos_weight=aux_pos_weight,
        final_threshold=0.5,
    )
    if predictions is None or "final_target" not in predictions:
        raise RuntimeError("Test predictions with final_target were not produced.")

    eval_df = predictions.dropna(subset=["final_target", "final_prob"]).copy()
    test_metadata = test_loader.dataset.df.reset_index(drop=True)
    patient_metadata = pd.DataFrame({
        base.IMAGE_COL: test_metadata[base.IMAGE_COL].astype(str),
        "patient_group": base._patient_groups(test_metadata, verbose=False).astype(str),
    }).drop_duplicates(subset=[base.IMAGE_COL])
    eval_df[base.IMAGE_COL] = eval_df[base.IMAGE_COL].astype(str)
    eval_df = eval_df.merge(patient_metadata, on=base.IMAGE_COL, how="left", validate="many_to_one")
    eval_df["patient_group"] = eval_df["patient_group"].fillna(eval_df[base.IMAGE_COL]).astype(str)
    y_true = eval_df["final_target"].astype(int).to_numpy()
    y_prob = eval_df["final_prob"].astype(float).to_numpy()

    fixed_thresholds = None
    validation_threshold_summary = pd.DataFrame()
    if cli_args.threshold_source == "validation":
        if val_predictions is None or "final_target" not in val_predictions:
            raise RuntimeError("Validation predictions are required for validation-fixed thresholds.")
        val_eval_df = val_predictions.dropna(subset=["final_target", "final_prob"]).copy()
        val_true = val_eval_df["final_target"].astype(int).to_numpy()
        val_prob = val_eval_df["final_prob"].astype(float).to_numpy()
        validation_threshold_summary = evaluate_threshold_modes(
            val_true,
            val_prob,
            target_specificity=cli_args.target_specificity,
            target_sensitivity=cli_args.target_sensitivity,
        )
        fixed_thresholds = {
            str(row.mode): float(row.threshold)
            for row in validation_threshold_summary.itertuples(index=False)
        }
        summary = evaluate_fixed_thresholds(y_true, y_prob, fixed_thresholds)
    else:
        summary = evaluate_threshold_modes(
            y_true,
            y_prob,
            target_specificity=cli_args.target_specificity,
            target_sensitivity=cli_args.target_sensitivity,
        )
    summary.insert(1, "threshold_source", cli_args.threshold_source)
    ci_summary, bootstrap_distribution = patient_cluster_bootstrap(
        eval_df,
        target_specificity=cli_args.target_specificity,
        target_sensitivity=cli_args.target_sensitivity,
        replicates=cli_args.bootstrap_replicates,
        confidence_level=cli_args.confidence_level,
        seed=cli_args.bootstrap_seed,
        show_progress=cli_args.show_progress,
        fixed_thresholds=fixed_thresholds,
    )
    if not ci_summary.empty:
        summary = summary.merge(ci_summary, on="mode", how="left", validate="one_to_one")
    for row in summary.itertuples(index=False):
        eval_df[f"pred_{row.mode}"] = (y_prob >= float(row.threshold)).astype(int)

    auxiliary_threshold_rows = []
    auxiliary_metrics_frames = [
        evaluate_auxiliary_features(
            eval_df,
            thresholds={task: 0.5 for task in base.AUX_COLUMNS},
            threshold_source="fixed_0.5",
        )
    ]
    if val_predictions is not None and "final_target" in val_predictions:
        validation_aux_thresholds = auxiliary_f1_thresholds(val_predictions)
        auxiliary_threshold_rows = [
            {
                "task": task,
                "threshold_source": "validation_f1",
                "threshold": threshold,
            }
            for task, threshold in validation_aux_thresholds.items()
        ]
        auxiliary_metrics_frames.append(
            evaluate_auxiliary_features(
                eval_df,
                thresholds=validation_aux_thresholds,
                threshold_source="validation_f1",
            )
        )
    auxiliary_metrics = pd.concat(auxiliary_metrics_frames, ignore_index=True)
    auxiliary_threshold_summary = pd.DataFrame(auxiliary_threshold_rows)

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing threshold metrics workbook -> {output_xlsx.resolve()}")
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        validation_threshold_summary.to_excel(writer, sheet_name="validation_thresholds", index=False)
        auxiliary_metrics.to_excel(writer, sheet_name="auxiliary_metrics", index=False)
        auxiliary_threshold_summary.to_excel(writer, sheet_name="auxiliary_thresholds", index=False)
        eval_df.to_excel(writer, sheet_name="test_predictions", index=False)
        ci_summary.to_excel(writer, sheet_name="bootstrap_ci", index=False)
        bootstrap_distribution.to_excel(writer, sheet_name="bootstrap_samples", index=False)
        pd.DataFrame([{
            "checkpoint": str(checkpoint_path),
            "stage": args.stage,
            "head_type": args.head_type,
            "test_auroc_from_validate": metrics.get("val_auroc_final"),
            "test_seg_valid_rate": metrics.get("val_seg_valid_rate"),
            "threshold_source": cli_args.threshold_source,
            "bootstrap_method": (
                "patient-cluster percentile bootstrap; validation threshold fixed"
                if fixed_thresholds is not None
                else "patient-cluster percentile bootstrap; test threshold re-estimated per replicate"
            ),
            "bootstrap_replicates_requested": cli_args.bootstrap_replicates,
            "bootstrap_seed": cli_args.bootstrap_seed,
            "confidence_level": cli_args.confidence_level,
            "patient_groups": int(eval_df["patient_group"].nunique()),
        }]).to_excel(writer, sheet_name="run_info", index=False)

    print(f"Saved Excel: {output_xlsx.resolve()}")
    print(summary.to_string(index=False))
    print("Auxiliary feature metrics:")
    print(auxiliary_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
