"""Zero-shot external validation for RG Graph-FiLM checkpoints.

Default checkpoint is the seed 1004 graph_film_latent_group_film model.
The model does not use segmentation priors, so external preprocessing is limited
to fundus bbox square crop + 224 resize + ImageNet normalization. A zero seg_input
is passed only to satisfy the shared model interface.

Supported datasets:
    REFUGE: labels inferred from Glaucoma / Non-Glaucoma folders.
    ORIGA: labels from origa_v2 sorted folders, metadata from glaucoma.csv if present.
    G1020: labels read from G1020.csv.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

try:
    from . import export_test_threshold_metrics as export_metrics
except ImportError:
    import export_test_threshold_metrics as export_metrics


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "outputs" / "external_validation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot external validation for graph_film_latent_group_film.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refuge-root", type=str, default=None)
    parser.add_argument("--origa-root", type=str, default=None)
    parser.add_argument("--g1020-root", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="refuge", help="Comma list: refuge,origa,g1020")
    parser.add_argument("--threshold-xlsx", type=str, default=None, help="Internal test_threshold_metrics.xlsx with validation thresholds.")
    parser.add_argument("--threshold-modes", type=str, default="fixed_0.5,youden,specificity_0.95,sensitivity_0.95,f1_optimal", help="Comma list of internal threshold modes to apply. Use sensitivity_0.95 for strict zero-shot Sens@95 internal threshold evaluation.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug limit per dataset split.")
    parser.add_argument("--save-crops", action="store_true", help="Save preprocessed crop224 images for inspection.")
    parser.add_argument("--crop-padding", type=float, default=0.02, help="Extra square bbox padding as fraction of bbox side.")
    parser.add_argument("--merge-origa-splits", action="store_true", help="Evaluate ORIGA Train and Validation together as split=all.")
    parser.add_argument("--merge-refuge-train-val", action="store_true", help="Evaluate REFUGE Training400 and Validation400 together as split=train_val; keep Test400 as split=test.")
    parser.add_argument("--include-external-oracle-threshold", action="store_true", help="Also report external-set F1-optimal threshold for exploratory analysis only.")
    return parser.parse_args()


def _scalar(value) -> object:
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    return value


def _as_str(value) -> str:
    value = _scalar(value)
    return str(value) if value is not None else ""


def _as_float(value) -> float:
    value = _scalar(value)
    try:
        return float(value)
    except Exception:
        return float("nan")


def collect_images_under(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def collect_refuge(root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    split_dirs = [("train", "Training400"), ("val", "Validation400"), ("test", "Test400")]
    for split, split_name in split_dirs:
        split_root = root / split_name
        if not split_root.exists():
            continue
        for label_dir, label in [("Glaucoma", 1), ("Non-Glaucoma", 0)]:
            for image_path in collect_images_under(split_root / label_dir):
                rows.append(
                    {
                        "dataset": "REFUGE",
                        "split": split,
                        "image_id": image_path.stem,
                        "image_path": str(image_path),
                        "label": int(label),
                    }
                )
    return pd.DataFrame(rows)


def collect_g1020(root: Path) -> pd.DataFrame:
    csv_path = root / "G1020.csv"
    image_root = root / "Images"
    if not csv_path.exists() or not image_root.exists():
        return pd.DataFrame()
    labels = pd.read_csv(csv_path)
    rows: List[dict] = []
    for _, row in labels.iterrows():
        filename = str(row.get("imageID", ""))
        image_path = image_root / filename
        if not image_path.exists():
            continue
        rows.append(
            {
                "dataset": "G1020",
                "split": "all",
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "label": int(row.get("binaryLabels")),
            }
        )
    return pd.DataFrame(rows)


def collect_origa(root: Path) -> pd.DataFrame:
    """Collect ORIGA v2 from sorted Train/Validation class folders.

    Expected root:
        E:\origa_v2
            glaucoma.csv
            Fundus_Train_Val_Data\Fundus_Scanes_Sorted\Train\Glaucoma_Negative
            Fundus_Train_Val_Data\Fundus_Scanes_Sorted\Train\Glaucoma_Positive
            Fundus_Train_Val_Data\Fundus_Scanes_Sorted\Validation\...

    Labels are taken from folder names to avoid accidentally using the cropped
    E:\ORIGA\Images copy. glaucoma.csv is used only for metadata sanity columns.
    """
    sorted_root = root / "Fundus_Train_Val_Data" / "Fundus_Scanes_Sorted"
    if not sorted_root.exists():
        # Backward-compatible fallback for the older ORIGA folder, but make it explicit.
        mat_path = root / "OrigaList.mat"
        image_root = root / "Images"
        if not mat_path.exists() or not image_root.exists():
            return pd.DataFrame()
        try:
            from scipy.io import loadmat
        except Exception as error:
            raise ImportError("scipy is required to parse ORIGA OrigaList.mat") from error
        mat = loadmat(mat_path)
        entries = mat.get("Origa")
        if entries is None:
            return pd.DataFrame()
        rows: List[dict] = []
        for entry in entries.reshape(-1):
            filename = _as_str(entry["Filename"])
            image_path = image_root / filename
            if not image_path.exists():
                continue
            rows.append(
                {
                    "dataset": "ORIGA",
                    "split": _as_str(entry["Set"]) or "all",
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "label": int(_as_float(entry["Glaucoma"])),
                    "eye": _as_str(entry["Eye"]),
                    "exp_cdr": _as_float(entry["ExpCDR"]),
                    "source_root": str(root),
                }
            )
        return pd.DataFrame(rows)

    metadata = pd.DataFrame()
    csv_path = root / "glaucoma.csv"
    if csv_path.exists():
        metadata = pd.read_csv(csv_path)
        metadata["Filename"] = metadata["Filename"].astype(str)

    meta_by_name = {
        str(row["Filename"]): row
        for _, row in metadata.iterrows()
    } if not metadata.empty and "Filename" in metadata.columns else {}

    rows: List[dict] = []
    split_map = {"Train": "train", "Validation": "val", "Val": "val", "Test": "test"}
    class_map = {"Glaucoma_Negative": 0, "Glaucoma_Positive": 1}
    for split_dir in sorted_root.iterdir():
        if not split_dir.is_dir():
            continue
        split = split_map.get(split_dir.name, split_dir.name.lower())
        for class_name, label in class_map.items():
            class_dir = split_dir / class_name
            for image_path in collect_images_under(class_dir):
                meta = meta_by_name.get(image_path.name)
                row = {
                    "dataset": "ORIGA",
                    "split": split,
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "label": int(label),
                    "source_root": str(root),
                    "source_class": class_name,
                }
                if meta is not None:
                    row["csv_label"] = int(meta["Glaucoma"]) if "Glaucoma" in meta and not pd.isna(meta["Glaucoma"]) else np.nan
                    row["eye"] = str(meta["Eye"]) if "Eye" in meta and not pd.isna(meta["Eye"]) else ""
                    row["exp_cdr"] = float(meta["ExpCDR"]) if "ExpCDR" in meta and not pd.isna(meta["ExpCDR"]) else np.nan
                    row["origa_set"] = str(meta["Set"]) if "Set" in meta and not pd.isna(meta["Set"]) else ""
                rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and "csv_label" in frame.columns:
        mismatched = frame.loc[frame["csv_label"].notna() & (frame["csv_label"].astype(int) != frame["label"].astype(int))]
        if not mismatched.empty:
            print(f"Warning: ORIGA folder label and glaucoma.csv mismatch for {len(mismatched)} image(s); using folder label.")
    return frame

def fundus_square_crop(image: Image.Image, output_size: int, padding_fraction: float = 0.02) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    threshold = max(8, int(np.percentile(blurred, 8)))
    mask = blurred > threshold
    # Remove tiny bright text/artifacts; keep the largest fundus component.
    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return image.convert("RGB").resize((output_size, output_size), Image.Resampling.BICUBIC)
    areas = stats[1:, cv2.CC_STAT_AREA]
    component = int(np.argmax(areas) + 1)
    ys, xs = np.where(labels == component)
    if xs.size == 0 or ys.size == 0:
        return image.convert("RGB").resize((output_size, output_size), Image.Resampling.BICUBIC)

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    h, w = rgb.shape[:2]
    side = max(x1 - x0, y1 - y0)
    side = int(round(side * (1.0 + max(0.0, padding_fraction))))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = left + side
    bottom = top + side

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        rgb = cv2.copyMakeBorder(rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    crop = rgb[top:bottom, left:right]
    return Image.fromarray(crop).resize((output_size, output_size), Image.Resampling.BICUBIC)


class ExternalFundusDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int, crop_padding: float, crop_output_dir: Optional[Path] = None) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.image_size = int(image_size)
        self.crop_padding = float(crop_padding)
        self.crop_output_dir = crop_output_dir
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        crop = fundus_square_crop(image, self.image_size, self.crop_padding)
        if self.crop_output_dir is not None:
            out_dir = self.crop_output_dir / str(row["dataset"]) / str(row["split"])
            out_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out_dir / f"{row['image_id']}.jpg", quality=95)
        image_tensor = TF.to_tensor(crop)
        image_tensor = (image_tensor - self.mean) / self.std
        return {
            "image": image_tensor,
            "seg_input": torch.zeros(2, self.image_size, self.image_size, dtype=torch.float32),
            "seg_valid": torch.tensor(1.0, dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }


def load_thresholds(
    threshold_xlsx: Optional[Path],
    checkpoint_path: Path,
    threshold_modes: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    if threshold_xlsx is None:
        candidate = checkpoint_path.parent / "test_threshold_metrics.xlsx"
        threshold_xlsx = candidate if candidate.exists() else None
    thresholds = {"fixed_0.5": 0.5}
    if threshold_xlsx is not None and threshold_xlsx.exists():
        summary = pd.read_excel(threshold_xlsx, sheet_name="summary")
        for _, row in summary.iterrows():
            mode = str(row.get("mode", ""))
            if mode:
                thresholds[mode] = float(row.get("threshold"))
    if threshold_modes:
        selected = {}
        missing = []
        for mode in threshold_modes:
            mode = str(mode).strip()
            if not mode:
                continue
            if mode not in thresholds:
                missing.append(mode)
            else:
                selected[mode] = thresholds[mode]
        if missing:
            available = ", ".join(sorted(thresholds.keys()))
            raise ValueError(f"Threshold mode(s) not found: {missing}. Available: {available}")
        thresholds = selected
    return thresholds


def build_model(checkpoint_path: Path, device: torch.device, batch_size: int, num_workers: int, amp: bool) -> Tuple[torch.nn.Module, argparse.Namespace]:
    cli = argparse.Namespace(batch_size=batch_size, num_workers=num_workers, device=str(device), disable_amp=not amp)
    args = export_metrics.load_run_args(checkpoint_path, cli)
    model = export_metrics.build_model(args, device)
    model.eval()
    return model, args


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Dict[str, float],
    include_external_oracle_threshold: bool = False,
) -> pd.DataFrame:
    rows: List[dict] = []
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid].astype(float)
    if y_true.size == 0:
        return pd.DataFrame()
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    eval_thresholds = dict(thresholds)
    if include_external_oracle_threshold and len(np.unique(y_true)) == 2:
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        f1_values = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
        best_idx = int(np.nanargmax(f1_values))
        if best_idx >= len(pr_thresholds):
            best_thr = 1.0
        else:
            best_thr = float(pr_thresholds[best_idx])
        eval_thresholds["external_f1_optimal_oracle"] = best_thr

    for mode, threshold in eval_thresholds.items():
        y_pred = (y_prob >= float(threshold)).astype(int)
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        else:
            tn = fp = fn = tp = 0
        precision_value = tp / max(tp + fp, 1)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        rows.append(
            {
                "mode": mode,
                "threshold": float(threshold),
                "n": int(y_true.size),
                "positive_n": int(y_true.sum()),
                "negative_n": int((1 - y_true).sum()),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(precision_value),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy": float((y_true == y_pred).mean()),
                "specificity": float(specificity),
                "sensitivity": float(sensitivity),
                "auroc": float(auroc),
                "auprc": float(auprc),
                "pred_pos_rate": float(y_pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def run_inference(model: torch.nn.Module, dataframe: pd.DataFrame, args: argparse.Namespace, output_dir: Path, save_crops: bool) -> pd.DataFrame:
    crop_dir = output_dir / "crop224" if save_crops else None
    dataset = ExternalFundusDataset(dataframe, image_size=args.image_size, crop_padding=args.crop_padding, crop_output_dir=crop_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = torch.cuda.is_available() and not args.disable_amp
    rows = dataframe.reset_index(drop=True).copy()
    probs = np.full(len(rows), np.nan, dtype=np.float32)
    rgb_probs = np.full(len(rows), np.nan, dtype=np.float32)
    aux_probs: List[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="External inference"):
            images = batch["image"].to(device, non_blocking=True)
            seg_input = batch["seg_input"].to(device, non_blocking=True)
            seg_valid = batch["seg_valid"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images, seg_input, seg_valid=seg_valid)
            batch_prob = torch.sigmoid(outputs["final_logits"]).detach().float().cpu().numpy()
            batch_rgb_prob = torch.sigmoid(outputs.get("rgb_base_logits", outputs["final_logits"])).detach().float().cpu().numpy()
            indices = batch["index"].numpy().astype(int)
            probs[indices] = batch_prob
            rgb_probs[indices] = batch_rgb_prob
            if "aux_logits" in outputs:
                aux = torch.sigmoid(outputs["aux_logits"]).detach().float().cpu().numpy()
                aux_probs.append(np.column_stack([indices, aux]))

    rows["prob_final"] = probs
    rows["prob_rgb_base"] = rgb_probs
    if aux_probs:
        aux_all = np.concatenate(aux_probs, axis=0)
        aux_all = aux_all[np.argsort(aux_all[:, 0])]
        aux_names = list(getattr(export_metrics.base, "AUX_COLUMNS", []))
        for idx, name in enumerate(aux_names):
            if idx + 1 < aux_all.shape[1]:
                rows[f"prob_aux_{name}"] = aux_all[:, idx + 1]
    return rows


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    selected = {name.strip().lower() for name in args.datasets.split(",") if name.strip()}
    frames: List[pd.DataFrame] = []
    if "refuge" in selected:
        frames.append(collect_refuge(Path(args.refuge_root)))
    if "origa" in selected:
        frames.append(collect_origa(Path(args.origa_root)))
    if "g1020" in selected:
        frames.append(collect_g1020(Path(args.g1020_root)))
    manifest = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if manifest.empty:
        raise FileNotFoundError("No external images found. Check --datasets and dataset roots.")
    if args.merge_origa_splits and not manifest.empty:
        is_origa = manifest["dataset"].astype(str).str.upper() == "ORIGA"
        manifest.loc[is_origa, "split"] = "all"
    if args.merge_refuge_train_val and not manifest.empty:
        is_refuge = manifest["dataset"].astype(str).str.upper() == "REFUGE"
        is_train_or_val = manifest["split"].astype(str).isin(["train", "val"])
        manifest.loc[is_refuge & is_train_or_val, "split"] = "train_val"
    if args.max_samples is not None:
        manifest = manifest.groupby(["dataset", "split"], group_keys=False).head(int(args.max_samples)).reset_index(drop=True)

    print("External manifest:")
    print(manifest.groupby(["dataset", "split", "label"]).size().rename("n").reset_index().to_string(index=False))
    manifest.to_csv(output_dir / "external_manifest.csv", index=False, encoding="utf-8-sig")

    model, run_args = build_model(checkpoint_path, device=device, batch_size=args.batch_size, num_workers=args.num_workers, amp=not args.disable_amp)
    threshold_modes = [mode.strip() for mode in str(args.threshold_modes).split(",") if mode.strip()]
    thresholds = load_thresholds(Path(args.threshold_xlsx) if args.threshold_xlsx else None, checkpoint_path, threshold_modes=threshold_modes)
    print("Fixed internal thresholds:", thresholds)

    predictions = run_inference(model, manifest, args, output_dir, save_crops=args.save_crops)
    predictions_path = output_dir / "external_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    metric_tables: Dict[str, pd.DataFrame] = {}
    summary_rows: List[pd.DataFrame] = []
    for (dataset_name, split_name), group in predictions.groupby(["dataset", "split"]):
        metrics = evaluate_predictions(
            group["label"].to_numpy(),
            group["prob_final"].to_numpy(),
            thresholds,
            include_external_oracle_threshold=bool(args.include_external_oracle_threshold),
        )
        if metrics.empty:
            continue
        external_name = f"{str(dataset_name).lower()}_{str(split_name).lower()}"
        if external_name == "refuge_train_val":
            external_name = "refuge_trainval"
        if external_name.endswith("_all") and str(dataset_name).upper() in {"ORIGA", "G1020"}:
            external_name = str(dataset_name).lower() if str(dataset_name).upper() == "G1020" else "origa_all"
        metrics.insert(0, "external_dataset", external_name)
        metrics.insert(1, "dataset", dataset_name)
        metrics.insert(2, "split", split_name)
        metric_tables[f"{dataset_name}_{split_name}"[:31]] = metrics
        summary_rows.append(metrics)
    all_metrics = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()

    xlsx_path = output_dir / "external_validation_metrics.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        predictions.to_excel(writer, sheet_name="predictions", index=False)
        all_metrics.to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame([{"checkpoint": str(checkpoint_path), "threshold_source": str(args.threshold_xlsx or checkpoint_path.parent / "test_threshold_metrics.xlsx")}]).to_excel(writer, sheet_name="run_info", index=False)
        for sheet, table in metric_tables.items():
            table.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved Excel: {xlsx_path}")
    if not all_metrics.empty:
        print(all_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
