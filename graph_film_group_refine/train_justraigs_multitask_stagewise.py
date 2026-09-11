import argparse
import copy
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_COL = "Image"
FINAL_COL = "Final"
PATIENT_GROUP_CANDIDATE_COLUMNS = [
    "patient_id",
    "PatientID",
    "patient",
    "Patient",
    "subject_id",
    "SubjectID",
    "case_id",
    "CaseID",
    "study_id",
    "StudyID",
]

ALL_AUX_COLUMNS = [
    "ANRS",
    "ANRI",
    "RNFLDS",
    "RNFLDI",
    "BCLVS",
    "BCLVI",
    "NVT",
    "DH",
    "LD",
    "LC",
]

ALL_MASK_COLUMNS = [f"{name}_m" for name in ALL_AUX_COLUMNS]

STAGE_TASKS = {
    "stage1": ["ANRS", "ANRI", "LD", "LC", "NVT"],
    "stage2": ["ANRS", "ANRI", "LD", "LC", "NVT", "BCLVS", "BCLVI", "RNFLDS", "RNFLDI"],
    "stage3": list(ALL_AUX_COLUMNS),
    "final_only": [],
}

AUX_COLUMNS = list(ALL_AUX_COLUMNS)
MASK_COLUMNS = [f"{name}_m" for name in AUX_COLUMNS]

DEFAULT_PROJECT_DIR = str(Path(__file__).resolve().parents[1])
DEFAULT_CSV_PATH = str(Path(DEFAULT_PROJECT_DIR) / "JustRAIGS_processed.csv")
DEFAULT_CACHE_DIR = str(Path(DEFAULT_PROJECT_DIR) / "data" / "cache")
DEFAULT_IMAGE_DIR = DEFAULT_CACHE_DIR
DEFAULT_OUTPUT_DIR = str(Path(DEFAULT_PROJECT_DIR) / "checkpoints")
DEFAULT_MODEL_NAME = "swin_tiny_patch4_window7_224"

GEOMETRY_FEATURE_COLUMNS = [
    "disc_area",
    "cup_area",
    "rim_area",
    "cdr_area",
    "rdr",
    "disc_width",
    "disc_height",
    "cup_width",
    "cup_height",
    "vcdr",
    "hcdr",
    "cup_eccentricity",
    "disc_cx",
    "disc_cy",
    "cup_cx",
    "cup_cy",
    "dx",
    "dy",
    "dx_norm",
    "dy_norm",
]

MODEL_ALIASES = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "convnext_tiny": "convnext_tiny",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _as_float(value: object) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def parse_binary_value(value) -> float:
    if pd.isna(value):
        return math.nan

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "nan", "none", "null"}:
            return math.nan
        if normalized in {"0", "false", "no", "n", "nrg", "normal"}:
            return 0.0
        if normalized in {"1", "true", "yes", "y", "rg", "glaucoma"}:
            return 1.0

    return float(value)


def parse_mask_value(value, fallback: bool) -> float:
    if pd.isna(value):
        return float(fallback)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "nan", "none", "null"}:
            return float(fallback)
        if normalized in {"0", "false", "no", "n"}:
            return 0.0
        if normalized in {"1", "true", "yes", "y"}:
            return 1.0

    return float(float(value) > 0)


def validate_columns(df: pd.DataFrame, require_targets: bool = True) -> None:
    required = [IMAGE_COL]
    if require_targets:
        required += [FINAL_COL, *AUX_COLUMNS, *MASK_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")


def merge_geometry_features(
    dataframe: pd.DataFrame,
    geometry_csv: Optional[str],
    geometry_feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not geometry_csv:
        return dataframe

    geometry_path = Path(geometry_csv)
    if not geometry_path.exists():
        raise FileNotFoundError(f"Geometry CSV not found: {geometry_path}")

    geometry_df = pd.read_csv(geometry_path)
    if IMAGE_COL not in geometry_df.columns:
        if "image_id" in geometry_df.columns:
            geometry_df = geometry_df.rename(columns={"image_id": IMAGE_COL})
        else:
            raise ValueError(f"Geometry CSV must contain '{IMAGE_COL}' or 'image_id' column: {geometry_path}")

    requested_columns = geometry_feature_columns or list(GEOMETRY_FEATURE_COLUMNS)
    missing = [column for column in requested_columns if column not in geometry_df.columns]
    if missing:
        raise ValueError(f"Geometry CSV is missing required feature columns: {missing}")

    geometry_subset = geometry_df[[IMAGE_COL, *requested_columns]].copy()
    merged = dataframe.merge(geometry_subset, on=IMAGE_COL, how="left")
    return merged


def compute_geometry_normalization_stats(
    train_df: pd.DataFrame,
    geometry_feature_columns: List[str],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for column in geometry_feature_columns:
        if column not in train_df.columns:
            continue
        series = pd.to_numeric(train_df[column], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            mean_value = 0.0
            std_value = 1.0
        else:
            mean_value = float(valid.mean())
            std_value = float(valid.std(ddof=0))
            if not np.isfinite(std_value) or std_value <= 1e-8:
                std_value = 1.0
        stats[column] = {"mean": mean_value, "std": std_value}
    return stats


def apply_geometry_normalization(
    dataframe: pd.DataFrame,
    geometry_feature_columns: List[str],
    stats: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    normalized_df = dataframe.copy()
    for column in geometry_feature_columns:
        if column not in normalized_df.columns or column not in stats:
            continue
        series = pd.to_numeric(normalized_df[column], errors="coerce")
        mean_value = float(stats[column]["mean"])
        std_value = float(stats[column]["std"])
        normalized_df[column] = (series - mean_value) / std_value
    return normalized_df


def log_geometry_normalization_stats(stats: Dict[str, Dict[str, float]]) -> None:
    if not stats:
        return
    summary = " | ".join(
        f"{column}: mean={values['mean']:.4f}, std={values['std']:.4f}"
        for column, values in stats.items()
    )
    print("Geometry normalization stats (train split) -> " + summary)


def configure_aux_tasks(selected_aux_columns: List[str]) -> None:
    global AUX_COLUMNS, MASK_COLUMNS

    invalid = [name for name in selected_aux_columns if name not in ALL_AUX_COLUMNS]
    if invalid:
        raise ValueError(f"Unknown auxiliary task names: {invalid}")

    deduped: List[str] = []
    seen = set()
    for name in selected_aux_columns:
        if name not in seen:
            seen.add(name)
            deduped.append(name)

    AUX_COLUMNS = deduped
    MASK_COLUMNS = [f"{name}_m" for name in AUX_COLUMNS]


def resolve_selected_aux_tasks(args: argparse.Namespace) -> List[str]:
    if args.aux_tasks:
        return [task.strip() for task in args.aux_tasks.split(",") if task.strip()]
    return list(STAGE_TASKS[args.stage])


def read_checkpoint_aux_columns(checkpoint_path: Path) -> Optional[List[str]]:
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    aux_columns = checkpoint.get("aux_columns")
    if not aux_columns:
        return None
    return [str(column) for column in aux_columns]


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


class NpyImageCache:
    def __init__(
        self,
        cache_dir: Path,
        images_name: str = "images.npy",
        paths_name: str = "paths.npy",
        mmap: bool = True,
        index_fallback: bool = True,
        lookup: str = "path",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.images_path = self.cache_dir / images_name
        self.paths_path = self.cache_dir / paths_name
        self.index_fallback = index_fallback
        self.mmap = mmap
        self.lookup = lookup

        if not self.images_path.exists():
            raise FileNotFoundError(f"Missing image cache file: {self.images_path}")
        if not self.paths_path.exists():
            raise FileNotFoundError(f"Missing path cache file: {self.paths_path}")

        self.images = self._open_images()
        self.paths = np.load(self.paths_path, allow_pickle=True)
        self.num_images = len(self.images)
        self.path_examples = [self._to_string(path) for path in self.paths[:5]]

        if self.num_images != len(self.paths):
            raise ValueError(
                f"images.npy and paths.npy length mismatch: {self.num_images} != {len(self.paths)}"
            )

        self.path_to_index = {} if lookup == "index" else self._build_path_index(self.paths)

    def _open_images(self) -> np.ndarray:
        mmap_mode = "r" if self.mmap else None
        return np.load(self.images_path, mmap_mode=mmap_mode)

    def _ensure_images(self) -> np.ndarray:
        if self.images is None:
            self.images = self._open_images()
        return self.images

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["images"] = None
        state["paths"] = None
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)

    @staticmethod
    def _to_string(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    @classmethod
    def _key_variants(cls, value) -> List[str]:
        raw = cls._to_string(value).strip()
        if not raw:
            return []

        normalized_path = os.path.normcase(os.path.normpath(raw))
        normalized_slash = normalized_path.replace("/", "\\")
        path_obj = Path(raw)

        candidates = [
            raw,
            raw.replace("/", "\\"),
            normalized_path,
            normalized_slash,
            path_obj.name,
            path_obj.stem,
            Path(normalized_slash).name,
            Path(normalized_slash).stem,
        ]

        deduped: List[str] = []
        seen = set()
        for candidate in candidates:
            key = str(candidate).strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

    @classmethod
    def _build_path_index(cls, paths: np.ndarray) -> Dict[str, Optional[int]]:
        path_to_index: Dict[str, Optional[int]] = {}
        for index, path_value in enumerate(paths):
            for key in cls._key_variants(path_value):
                if key in path_to_index:
                    path_to_index[key] = None
                else:
                    path_to_index[key] = index
        return path_to_index

    def get(self, image_value, source_index: Optional[int] = None) -> np.ndarray:
        images = self._ensure_images()

        if self.lookup == "index":
            if source_index is None:
                raise KeyError("cache lookup is set to 'index', but source_index is missing.")
            if 0 <= int(source_index) < self.num_images:
                return images[int(source_index)]
            raise IndexError(f"source_index {source_index} is outside images.npy length {self.num_images}")

        for key in self._key_variants(image_value):
            cache_index = self.path_to_index.get(key)
            if cache_index is not None:
                return images[int(cache_index)]

        if self.index_fallback and source_index is not None and 0 <= int(source_index) < self.num_images:
            return images[int(source_index)]

        raise KeyError(
            f"Could not match CSV Image value '{image_value}' to paths.npy. "
            f"First cached paths: {self.path_examples}"
        )


class JustRAIGSMultiTaskDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Optional[Path] = None,
        image_cache: Optional[NpyImageCache] = None,
        transform: Optional[transforms.Compose] = None,
        require_targets: bool = True,
        geometry_feature_columns: Optional[List[str]] = None,
    ) -> None:
        validate_columns(dataframe, require_targets=require_targets)
        self.source_indices = dataframe.index.to_numpy()
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.image_cache = image_cache
        self.transform = transform
        self.require_targets = require_targets
        self.geometry_feature_columns = geometry_feature_columns or []

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_value) -> Path:
        if self.image_dir is None:
            raise ValueError("image_dir is required when image_cache is not used.")

        raw_path = Path(str(image_value))
        candidates: List[Path] = []

        if raw_path.is_absolute():
            candidates.append(raw_path)
            candidates.append(raw_path.with_suffix(".npy"))
            candidates.append(raw_path.parent / f"{raw_path.name}.npy")
        else:
            candidates.append(self.image_dir / raw_path)
            candidates.append(self.image_dir / raw_path.with_suffix(".npy"))
            candidates.append(self.image_dir / f"{raw_path.name}.npy")
            candidates.append(self.image_dir / raw_path.name)

        seen = set()
        for candidate in candidates:
            normalized = candidate.resolve()
            if normalized in seen:
                continue
            seen.add(normalized)
            if normalized.exists():
                return normalized

        raise FileNotFoundError(
            f"Image not found for CSV value '{image_value}'. Tried: {[str(path) for path in candidates]}"
        )

    @staticmethod
    def _array_to_rgb_image(array: np.ndarray) -> Image.Image:
        array = np.asarray(array)

        if array.ndim == 2:
            array = np.repeat(array[..., None], repeats=3, axis=-1)
        elif array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3, 4}:
            array = np.transpose(array, (1, 2, 0))

        if array.ndim != 3:
            raise ValueError(f"Expected 2D or 3D image array, got shape {array.shape}")

        if array.shape[-1] == 1:
            array = np.repeat(array, repeats=3, axis=-1)
        elif array.shape[-1] > 3:
            array = array[..., :3]
        elif array.shape[-1] != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got shape {array.shape}")

        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 1.0
            if max_value <= 1.0:
                array = array * 255.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)

        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    def _load_image(self, image_path: Path) -> Image.Image:
        if image_path.suffix.lower() == ".npy":
            image_array = np.load(image_path)
            return self._array_to_rgb_image(image_array)

        return Image.open(image_path).convert("RGB")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        item: Dict[str, torch.Tensor] = {"image_id": row[IMAGE_COL]}

        if self.image_cache is not None:
            image_array = self.image_cache.get(row[IMAGE_COL], source_index=int(self.source_indices[index]))
            image = self._array_to_rgb_image(image_array)
        else:
            image_path = self._resolve_image_path(row[IMAGE_COL])
            image = self._load_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        item["image"] = image

        if self.geometry_feature_columns:
            geometry_values: List[float] = []
            geometry_masks: List[float] = []
            for column in self.geometry_feature_columns:
                raw_value = row[column] if column in row.index else math.nan
                if pd.isna(raw_value):
                    geometry_values.append(0.0)
                    geometry_masks.append(0.0)
                else:
                    geometry_values.append(float(raw_value))
                    geometry_masks.append(1.0)
            item["geometry"] = torch.tensor(geometry_values, dtype=torch.float32)
            item["geometry_mask"] = torch.tensor(geometry_masks, dtype=torch.float32)

        if not self.require_targets:
            return item  # type: ignore[return-value]

        final_label = parse_binary_value(row[FINAL_COL])
        if math.isnan(final_label):
            raise ValueError(f"Missing final label at dataset index {index}")

        aux_labels: List[float] = []
        aux_masks: List[float] = []
        for label_col, mask_col in zip(AUX_COLUMNS, MASK_COLUMNS):
            raw_label = parse_binary_value(row[label_col])
            has_label = not math.isnan(raw_label)
            mask = parse_mask_value(row[mask_col], fallback=has_label) if has_label else 0.0

            aux_labels.append(0.0 if math.isnan(raw_label) else raw_label)
            aux_masks.append(mask)

        item["final"] = torch.tensor(final_label, dtype=torch.float32)
        item["aux"] = torch.tensor(aux_labels, dtype=torch.float32)
        item["aux_mask"] = torch.tensor(aux_masks, dtype=torch.float32)
        return item  # type: ignore[return-value]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _load_image_backend(args: argparse.Namespace) -> Tuple[Path, Optional[NpyImageCache]]:
    csv_parent = Path(args.csv).resolve().parent
    image_dir = Path(args.image_dir).resolve() if args.image_dir else csv_parent
    image_cache = None

    if not args.disable_npy_cache and args.cache_dir:
        cache_dir = Path(args.cache_dir).resolve()
        images_path = cache_dir / args.cache_images_name
        paths_path = cache_dir / args.cache_paths_name

        if images_path.exists() and paths_path.exists():
            image_cache = NpyImageCache(
                cache_dir=cache_dir,
                images_name=args.cache_images_name,
                paths_name=args.cache_paths_name,
                mmap=not args.disable_cache_mmap,
                index_fallback=not args.disable_cache_index_fallback,
                lookup=args.cache_lookup,
            )
            print(f"Using npy image cache: {images_path}")
        else:
            print(f"Npy cache not found at {cache_dir}; falling back to image files under {image_dir}")

    return image_dir, image_cache


def _make_loader(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
    image_dir: Path,
    image_cache: Optional[NpyImageCache],
    train: bool,
    require_targets: bool = True,
) -> DataLoader:
    dataset = JustRAIGSMultiTaskDataset(
        dataframe=dataframe,
        image_dir=image_dir,
        image_cache=image_cache,
        transform=build_transforms(args.image_size, train=train),
        require_targets=require_targets,
        geometry_feature_columns=getattr(args, "geometry_feature_columns", []),
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    sampler = None
    shuffle = train
    if train and require_targets and args.balanced_final_sampler and not args.disable_balanced_final_sampler:
        labels = _require_binary_labels(dataframe)
        class_counts = labels.value_counts().to_dict()
        class_weights = {label: 1.0 / max(count, 1) for label, count in class_counts.items()}
        sample_weights = labels.map(class_weights).astype(float).to_numpy()
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
        print(f"Using balanced final-label sampler. Final class counts: {class_counts}")

    return DataLoader(dataset, shuffle=shuffle, sampler=sampler, drop_last=train, **loader_kwargs)


def _require_binary_labels(df: pd.DataFrame) -> pd.Series:
    labels = df[FINAL_COL].apply(parse_binary_value)
    if labels.isna().any():
        bad_count = int(labels.isna().sum())
        raise ValueError(f"{bad_count} rows have missing or invalid final labels.")
    return labels.astype(int)


def _derive_patient_group_from_image_id(image_id: object) -> str:
    stem = Path(str(image_id)).stem
    # Common fundus naming patterns use a patient/study id plus eye/view suffix.
    for pattern in [
        r"(?i)(?:[_\-. ](?:left|right|os|od|ou|le|re|l|r))$",
        r"(?i)(?:[_\-. ]eye(?:[_\-. ]?[lr]))$",
        r"(?i)(?:[_\-. ](?:fundus|disc|macula)\d*)$",
    ]:
        stem = re.sub(pattern, "", stem)
    return stem or str(image_id)


def _patient_groups(dataframe: pd.DataFrame, verbose: bool = True) -> pd.Series:
    for column in PATIENT_GROUP_CANDIDATE_COLUMNS:
        if column in dataframe.columns:
            groups = dataframe[column].fillna(dataframe[IMAGE_COL]).astype(str)
            if verbose:
                print(f"Patient-level split group column -> {column}")
            return groups

    groups = dataframe[IMAGE_COL].map(_derive_patient_group_from_image_id).astype(str)
    unique_groups = int(groups.nunique())
    if verbose:
        if unique_groups == len(dataframe):
            print(
                "Patient-level split requested by default, but no patient column or repeated derived Image groups were found. "
                "Using one Image per patient group."
            )
        else:
            print(f"Patient-level split groups derived from Image -> groups={unique_groups}, images={len(dataframe)}")
    return groups


def _stratified_group_split(
    dataframe: pd.DataFrame,
    labels: pd.Series,
    split_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split_size <= 0 or split_size >= 1:
        raise ValueError(f"split_size must be between 0 and 1, got {split_size}")

    class_counts = labels.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Stratified split requires at least 2 samples in each class. "
            f"Class counts: {class_counts.to_dict()}"
        )

    working = dataframe.copy()
    working["_split_label"] = labels.to_numpy()
    working["_split_group"] = _patient_groups(working).to_numpy()
    group_summary = (
        working.groupby("_split_group", sort=False)
        .agg(label=("_split_label", "max"), n=("_split_label", "size"))
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    right_groups: set[str] = set()
    for label_value, label_df in group_summary.groupby("label", sort=True):
        label_groups = label_df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        target_count = max(1, int(round(float(label_df["n"].sum()) * split_size)))
        selected_count = 0
        for _, row in label_groups.iterrows():
            if selected_count >= target_count and selected_count > 0:
                break
            right_groups.add(str(row["_split_group"]))
            selected_count += int(row["n"])

    right_mask = working["_split_group"].astype(str).isin(right_groups)
    left_df = dataframe.loc[~right_mask].copy()
    right_df = dataframe.loc[right_mask].copy()
    if left_df.empty or right_df.empty:
        raise ValueError("Patient-level group split produced an empty split. Check class counts and group ids.")

    left_groups = _patient_groups(left_df, verbose=False)
    right_groups = _patient_groups(right_df, verbose=False)
    overlap = set(left_groups) & set(right_groups)
    if overlap:
        raise RuntimeError(f"Patient-level split leakage detected: {len(overlap)} group(s) overlap.")

    print(
        "Patient-level split -> "
        f"left_images={len(left_df)}, right_images={len(right_df)}, "
        f"left_groups={left_groups.nunique()}, right_groups={right_groups.nunique()}"
    )
    return left_df, right_df


def build_train_val_test_dataframes(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_df = pd.read_csv(args.csv)
    validate_columns(full_df, require_targets=True)
    full_df = merge_geometry_features(full_df, args.geometry_csv, getattr(args, "geometry_feature_columns", None))
    labels = _require_binary_labels(full_df)
    split_manifest_value = getattr(args, "split_manifest", None)
    split_manifest_path = Path(split_manifest_value) if split_manifest_value else None

    if split_manifest_path is not None and split_manifest_path.exists():
        manifest = pd.read_csv(split_manifest_path)
        required_manifest_columns = {IMAGE_COL, "split"}
        missing_columns = required_manifest_columns - set(manifest.columns)
        if missing_columns:
            raise ValueError(
                f"Split manifest is missing columns {sorted(missing_columns)}: {split_manifest_path}"
            )
        manifest = manifest[[IMAGE_COL, "split"]].copy()
        manifest[IMAGE_COL] = manifest[IMAGE_COL].astype(str)
        manifest["split"] = manifest["split"].astype(str).str.lower()
        if manifest[IMAGE_COL].duplicated().any():
            raise ValueError(f"Split manifest contains duplicate Image ids: {split_manifest_path}")
        invalid_splits = sorted(set(manifest["split"]) - {"train", "val", "test"})
        if invalid_splits:
            raise ValueError(f"Split manifest contains invalid split names: {invalid_splits}")

        full_ids = full_df[IMAGE_COL].astype(str)
        split_lookup = manifest.set_index(IMAGE_COL)["split"]
        assigned_splits = full_ids.map(split_lookup)
        if assigned_splits.isna().any():
            missing_ids = full_ids[assigned_splits.isna()].head(10).tolist()
            raise ValueError(
                f"Split manifest does not cover {int(assigned_splits.isna().sum())} dataset images. "
                f"Examples: {missing_ids}"
            )
        unknown_ids = sorted(set(manifest[IMAGE_COL]) - set(full_ids))
        if unknown_ids:
            raise ValueError(
                f"Split manifest contains {len(unknown_ids)} images absent from the dataset. "
                f"Examples: {unknown_ids[:10]}"
            )

        train_df = full_df.loc[assigned_splits.to_numpy() == "train"].copy()
        val_df = full_df.loc[assigned_splits.to_numpy() == "val"].copy()
        test_df = full_df.loc[assigned_splits.to_numpy() == "test"].copy()
        if min(len(train_df), len(val_df), len(test_df)) == 0:
            raise ValueError(f"Split manifest produced an empty split: {split_manifest_path}")
        print(
            f"Using fixed split manifest -> {split_manifest_path} | "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        if args.geometry_feature_columns and args.geometry_normalize:
            args.geometry_norm_stats = compute_geometry_normalization_stats(train_df, args.geometry_feature_columns)
            log_geometry_normalization_stats(args.geometry_norm_stats)
            train_df = apply_geometry_normalization(train_df, args.geometry_feature_columns, args.geometry_norm_stats)
            val_df = apply_geometry_normalization(val_df, args.geometry_feature_columns, args.geometry_norm_stats)
            test_df = apply_geometry_normalization(test_df, args.geometry_feature_columns, args.geometry_norm_stats)
        return train_df, val_df, test_df

    if args.test_size <= 0 or args.test_size >= 1:
        raise ValueError(f"--test-size must be between 0 and 1, got {args.test_size}")
    if args.val_size <= 0 or args.val_size >= 1:
        raise ValueError(f"--val-size must be between 0 and 1, got {args.val_size}")
    if args.val_size + args.test_size >= 1:
        raise ValueError("--val-size + --test-size must be less than 1.")

    train_val_df, test_df = _stratified_group_split(full_df, labels, split_size=args.test_size, seed=args.seed)

    if args.val_csv:
        val_df = pd.read_csv(args.val_csv)
        validate_columns(val_df, require_targets=True)
        val_df = merge_geometry_features(val_df, args.geometry_csv, getattr(args, "geometry_feature_columns", None))
        train_df = train_val_df
        if args.geometry_feature_columns and args.geometry_normalize:
            args.geometry_norm_stats = compute_geometry_normalization_stats(train_df, args.geometry_feature_columns)
            log_geometry_normalization_stats(args.geometry_norm_stats)
            train_df = apply_geometry_normalization(train_df, args.geometry_feature_columns, args.geometry_norm_stats)
            val_df = apply_geometry_normalization(val_df, args.geometry_feature_columns, args.geometry_norm_stats)
            test_df = apply_geometry_normalization(test_df, args.geometry_feature_columns, args.geometry_norm_stats)
        return train_df, val_df, test_df

    val_fraction_of_remaining = args.val_size / (1.0 - args.test_size)
    train_df, val_df = _stratified_group_split(
        train_val_df,
        _require_binary_labels(train_val_df),
        split_size=val_fraction_of_remaining,
        seed=args.seed,
    )
    if split_manifest_path is not None:
        if full_df[IMAGE_COL].astype(str).duplicated().any():
            raise ValueError("Cannot write split manifest because the dataset contains duplicate Image ids.")
        split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = pd.concat(
            [
                pd.DataFrame({IMAGE_COL: train_df[IMAGE_COL].astype(str), "split": "train"}),
                pd.DataFrame({IMAGE_COL: val_df[IMAGE_COL].astype(str), "split": "val"}),
                pd.DataFrame({IMAGE_COL: test_df[IMAGE_COL].astype(str), "split": "test"}),
            ],
            ignore_index=True,
        )
        manifest.to_csv(split_manifest_path, index=False)
        print(
            f"Saved fixed split manifest -> {split_manifest_path} | "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    if args.geometry_feature_columns and args.geometry_normalize:
        args.geometry_norm_stats = compute_geometry_normalization_stats(train_df, args.geometry_feature_columns)
        log_geometry_normalization_stats(args.geometry_norm_stats)
        train_df = apply_geometry_normalization(train_df, args.geometry_feature_columns, args.geometry_norm_stats)
        val_df = apply_geometry_normalization(val_df, args.geometry_feature_columns, args.geometry_norm_stats)
        test_df = apply_geometry_normalization(test_df, args.geometry_feature_columns, args.geometry_norm_stats)
    return train_df, val_df, test_df


def compute_aux_pos_weights(train_df: pd.DataFrame, max_weight: float = 10.0, power: float = 0.5) -> torch.Tensor:
    if not AUX_COLUMNS:
        print("Auxiliary pos_weight skipped -> no active auxiliary tasks.")
        return torch.empty(0, dtype=torch.float32)

    weights: List[float] = []
    summaries: List[str] = []

    for label_col, mask_col in zip(AUX_COLUMNS, MASK_COLUMNS):
        labels = train_df[label_col].apply(parse_binary_value)
        valid_label = ~labels.isna()
        masks = pd.Series(0.0, index=train_df.index, dtype=float)
        masks.loc[valid_label] = train_df.loc[valid_label, mask_col].apply(
            lambda value: parse_mask_value(value, fallback=True)
        )
        valid = valid_label & (masks > 0)

        positives = int((labels[valid] == 1).sum())
        negatives = int((labels[valid] == 0).sum())

        if positives <= 0:
            weight = 1.0
        else:
            weight = (negatives / max(positives, 1)) ** power
            weight = min(float(weight), float(max_weight))

        weights.append(float(weight))
        summaries.append(f"{label_col}: pos={positives}, neg={negatives}, pos_weight={weight:.2f}")

    print("Auxiliary pos_weight from train split -> " + " | ".join(summaries))
    return torch.tensor(weights, dtype=torch.float32)


def compute_final_pos_weight(train_df: pd.DataFrame, max_weight: float = 10.0, power: float = 0.5) -> torch.Tensor:
    labels = _require_binary_labels(train_df)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())

    if positives <= 0:
        weight = 1.0
    else:
        weight = min(float((negatives / max(positives, 1)) ** power), float(max_weight))

    print(f"Final pos_weight from train split -> pos={positives}, neg={negatives}, pos_weight={weight:.2f}")
    return torch.tensor(weight, dtype=torch.float32)


def _safe_logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-4), 1.0 - 1e-4)
    return math.log(probability / (1.0 - probability))


def compute_train_label_priors(train_df: pd.DataFrame) -> Tuple[float, torch.Tensor]:
    final_labels = _require_binary_labels(train_df)
    final_prior = float(final_labels.mean())

    if not AUX_COLUMNS:
        print(f"Label priors from train split -> Final={final_prior:.4f} | no active auxiliary tasks")
        return final_prior, torch.empty(0, dtype=torch.float32)

    aux_priors: List[float] = []
    for label_col, mask_col in zip(AUX_COLUMNS, MASK_COLUMNS):
        labels = train_df[label_col].apply(parse_binary_value)
        valid_label = ~labels.isna()
        masks = pd.Series(0.0, index=train_df.index, dtype=float)
        masks.loc[valid_label] = train_df.loc[valid_label, mask_col].apply(
            lambda value: parse_mask_value(value, fallback=True)
        )
        valid = valid_label & (masks > 0)
        aux_priors.append(float(labels[valid].mean()) if valid.any() else 0.5)

    print(
        f"Label priors from train split -> Final={final_prior:.4f} | "
        + " | ".join(f"{name}={prior:.4f}" for name, prior in zip(AUX_COLUMNS, aux_priors))
    )
    return final_prior, torch.tensor(aux_priors, dtype=torch.float32)


def initialize_output_biases(
    model: nn.Module,
    final_prior: float,
    aux_priors: torch.Tensor,
) -> None:
    final_linear = model.final_head[-1]
    aux_linear = model.aux_head[-1]
    if not isinstance(final_linear, nn.Linear) or not isinstance(aux_linear, nn.Linear):
        raise TypeError("Expected final_head and aux_head to end with nn.Linear.")

    with torch.no_grad():
        final_linear.bias.fill_(_safe_logit(final_prior))
        if aux_linear.bias.numel() > 0:
            aux_bias = torch.tensor([_safe_logit(float(prior)) for prior in aux_priors], dtype=aux_linear.bias.dtype)
            aux_linear.bias.copy_(aux_bias.to(aux_linear.bias.device))


def build_eval_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.eval_csv:
        eval_df = pd.read_csv(args.eval_csv)
        validate_columns(eval_df, require_targets=not args.allow_unlabeled_eval)
        eval_df = merge_geometry_features(eval_df, args.geometry_csv, getattr(args, "geometry_feature_columns", None))
        if args.geometry_feature_columns and args.geometry_normalize and getattr(args, "geometry_norm_stats", None):
            eval_df = apply_geometry_normalization(eval_df, args.geometry_feature_columns, args.geometry_norm_stats)
        return eval_df

    _, _, test_df = build_train_val_test_dataframes(args)
    validate_columns(test_df, require_targets=not args.allow_unlabeled_eval)
    return test_df


def build_loaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    train_df, val_df, test_df = build_train_val_test_dataframes(args)
    image_dir, image_cache = _load_image_backend(args)
    final_pos_weight = compute_final_pos_weight(
        train_df,
        max_weight=args.final_pos_weight_max,
        power=args.pos_weight_power,
    )
    aux_pos_weight = compute_aux_pos_weights(
        train_df,
        max_weight=args.aux_pos_weight_max,
        power=args.pos_weight_power,
    )
    final_prior, aux_priors = compute_train_label_priors(train_df)

    train_loader = _make_loader(train_df, args, image_dir, image_cache, train=True, require_targets=True)
    val_loader = _make_loader(val_df, args, image_dir, image_cache, train=False, require_targets=True)
    test_loader = _make_loader(test_df, args, image_dir, image_cache, train=False, require_targets=True)
    print(
        f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} "
        f"(RG/NRG stratified on Final)"
    )
    return train_loader, val_loader, test_loader, final_pos_weight, aux_pos_weight, final_prior, aux_priors


def build_eval_loader(args: argparse.Namespace) -> DataLoader:
    eval_df = build_eval_dataframe(args)
    image_dir, image_cache = _load_image_backend(args)
    return _make_loader(
        eval_df,
        args,
        image_dir,
        image_cache,
        train=False,
        require_targets=not args.allow_unlabeled_eval,
    )


def _stage_output_dir(base_output_dir: str, stage: str) -> Path:
    output_dir = Path(base_output_dir)
    if Path(base_output_dir) == Path(DEFAULT_OUTPUT_DIR):
        output_dir = output_dir / f"stage_runs_{stage}"
    return output_dir


class MixtureOfExpertsBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 4,
        hidden_features: Optional[int] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        self.num_experts = num_experts

        self.gate = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_experts),
        )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Linear(in_features, hidden_features),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_features, out_features),
                    nn.GELU(),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)
        return mixed, gate_probs


class JustRAIGSConvNeXtMoE(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: bool = False,
        aux_tasks: int = len(AUX_COLUMNS),
        num_experts: int = 4,
        moe_dim: int = 512,
        dropout: float = 0.2,
        head_type: str = "simple",
        geometry_dim: int = 0,
        geometry_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        model_name = resolve_model_name(model_name)
        if head_type not in {"simple", "moe"}:
            raise ValueError(f"Unsupported head_type: {head_type}")
        self.head_type = head_type

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        backbone_features = self.backbone.num_features

        if head_type == "moe":
            self.feature_head = MixtureOfExpertsBlock(
                in_features=backbone_features,
                out_features=moe_dim,
                num_experts=num_experts,
                hidden_features=backbone_features,
                dropout=dropout,
            )
        else:
            self.feature_head = nn.Sequential(
                nn.LayerNorm(backbone_features),
                nn.Linear(backbone_features, moe_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.geometry_dim = int(geometry_dim)
        self.geometry_hidden_dim = int(geometry_hidden_dim)
        if self.geometry_dim > 0:
            self.geometry_encoder = nn.Sequential(
                nn.LayerNorm(self.geometry_dim),
                nn.Linear(self.geometry_dim, self.geometry_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_input_dim = moe_dim + self.geometry_hidden_dim
        else:
            self.geometry_encoder = None
            head_input_dim = moe_dim

        self.final_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim, 1),
        )

        self.aux_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim, aux_tasks),
        )

    @staticmethod
    def _tokens_to_spatial_map(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor with shape [B, L, C], got {tuple(tokens.shape)}")
        batch_size, num_tokens, channels = tokens.shape
        side = int(math.sqrt(num_tokens))
        if side * side == num_tokens:
            return tokens.transpose(1, 2).reshape(batch_size, channels, side, side)
        if num_tokens > 1:
            side = int(math.sqrt(num_tokens - 1))
            if side * side == (num_tokens - 1):
                tokens_wo_cls = tokens[:, 1:, :]
                return tokens_wo_cls.transpose(1, 2).reshape(batch_size, channels, side, side)
        raise ValueError(f"Could not reshape token tensor of length {num_tokens} into a spatial map.")

    def _backbone_forward_spatial(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(images)
        else:
            features = self.backbone(images)

        if features.ndim == 4:
            if features.shape[1] == self.backbone.num_features:
                return features
            if features.shape[-1] == self.backbone.num_features:
                return features.permute(0, 3, 1, 2).contiguous()
        if features.ndim == 3:
            return self._tokens_to_spatial_map(features)
        raise ValueError(f"Unsupported backbone feature shape for Grad-CAM: {tuple(features.shape)}")

    @staticmethod
    def _pool_spatial_features(spatial_features: torch.Tensor) -> torch.Tensor:
        return spatial_features.mean(dim=(2, 3))

    def _forward_heads_from_pooled(
        self,
        pooled_features: torch.Tensor,
        geometry: Optional[torch.Tensor] = None,
        geometry_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.head_type == "moe":
            head_features, gate_probs = self.feature_head(pooled_features)
        else:
            head_features = self.feature_head(pooled_features)
            gate_probs = None

        if self.geometry_encoder is not None:
            if geometry is None:
                geometry = head_features.new_zeros((head_features.size(0), self.geometry_dim))
            if geometry_mask is not None:
                geometry = geometry * geometry_mask
            geometry_features = self.geometry_encoder(geometry)
            head_features = torch.cat([head_features, geometry_features], dim=1)

        final_logits = self.final_head(head_features).squeeze(1)
        aux_logits = self.aux_head(head_features)
        return {
            "final_logits": final_logits,
            "aux_logits": aux_logits,
            "gate_probs": gate_probs,
            "head_features": head_features,
        }

    def forward(
        self,
        images: torch.Tensor,
        geometry: Optional[torch.Tensor] = None,
        geometry_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        spatial_features = self._backbone_forward_spatial(images)
        pooled_features = self._pool_spatial_features(spatial_features)
        outputs = self._forward_heads_from_pooled(
            pooled_features,
            geometry=geometry,
            geometry_mask=geometry_mask,
        )
        outputs["spatial_features"] = spatial_features
        return outputs


def build_optimizer(
    model: JustRAIGSConvNeXtMoE,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    parameter_groups = [
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.feature_head.parameters(), "lr": head_lr},
        {"params": model.final_head.parameters(), "lr": head_lr},
        {"params": model.aux_head.parameters(), "lr": head_lr},
    ]
    if getattr(model, "geometry_encoder", None) is not None:
        parameter_groups.append({"params": model.geometry_encoder.parameters(), "lr": head_lr})

    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def masked_auxiliary_loss(
    aux_logits: torch.Tensor,
    aux_targets: torch.Tensor,
    aux_masks: torch.Tensor,
    aux_pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if aux_logits.numel() == 0 or aux_targets.numel() == 0 or aux_masks.numel() == 0:
        return aux_logits.new_zeros(())

    raw_loss = F.binary_cross_entropy_with_logits(
        aux_logits,
        aux_targets,
        reduction="none",
        pos_weight=aux_pos_weight,
    )
    masked_loss = raw_loss * aux_masks
    denominator = aux_masks.sum().clamp_min(1.0)
    return masked_loss.sum() / denominator


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    final_pos_weight: Optional[torch.Tensor] = None,
    aux_pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    final_loss = F.binary_cross_entropy_with_logits(
        outputs["final_logits"],
        batch["final"],
        pos_weight=final_pos_weight,
    )
    aux_loss = masked_auxiliary_loss(
        outputs["aux_logits"],
        batch["aux"],
        batch["aux_mask"],
        aux_pos_weight=aux_pos_weight,
    )
    total_loss = final_loss + 0.5 * aux_loss

    return {
        "loss": total_loss,
        "loss_final": final_loss,
        "loss_aux": aux_loss,
    }


def collect_probability_summary(
    predictions: Optional[pd.DataFrame],
    threshold: float = 0.5,
) -> Dict[str, float]:
    if predictions is None or predictions.empty or "final_prob" not in predictions.columns:
        return {}

    final_probs = predictions["final_prob"].astype(float).to_numpy()
    summary = {
        "val_prob_mean_final": float(np.mean(final_probs)),
        "val_prob_std_final": float(np.std(final_probs)),
        "val_pred_pos_rate_final": float(np.mean(final_probs >= threshold)),
    }

    for aux_name in AUX_COLUMNS:
        prob_col = f"{aux_name}_prob"
        mask_col = f"{aux_name}_mask"
        if prob_col not in predictions.columns:
            continue

        if mask_col in predictions.columns:
            frame = predictions[predictions[mask_col].astype(float) > 0]
        else:
            frame = predictions
        if frame.empty:
            continue

        aux_probs = frame[prob_col].astype(float).to_numpy()
        summary[f"val_prob_mean_{aux_name}"] = float(np.mean(aux_probs))
        summary[f"val_prob_std_{aux_name}"] = float(np.std(aux_probs))
        summary[f"val_pred_pos_rate_{aux_name}"] = float(np.mean(aux_probs >= threshold))

    return summary


def rename_metric_prefix(metrics: Dict[str, float], old_prefix: str, new_prefix: str) -> Dict[str, float]:
    renamed: Dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith(old_prefix):
            renamed[new_prefix + key[len(old_prefix):]] = value
        else:
            renamed[key] = value
    return renamed


def save_metrics_history(rows: List[Dict[str, float]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def safe_auroc(targets: Iterable[float], probabilities: Iterable[float]) -> float:
    targets_array = np.asarray(list(targets), dtype=np.float32)
    probs_array = np.asarray(list(probabilities), dtype=np.float32)

    if targets_array.size == 0 or len(np.unique(targets_array)) < 2:
        return float("nan")

    return float(roc_auc_score(targets_array, probs_array))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    grad_clip_norm: Optional[float],
    final_pos_weight: Optional[torch.Tensor],
    aux_pos_weight: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.train()

    loss_meter = AverageMeter()
    final_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    progress = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(
                batch["image"],
                geometry=batch.get("geometry"),
                geometry_mask=batch.get("geometry_mask"),
            )
            losses = compute_loss(
                outputs,
                batch,
                final_pos_weight=final_pos_weight,
                aux_pos_weight=aux_pos_weight,
            )

        scaler.scale(losses["loss"]).backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["image"].size(0)
        loss_meter.update(float(losses["loss"].detach()), batch_size)
        final_loss_meter.update(float(losses["loss_final"].detach()), batch_size)
        aux_loss_meter.update(float(losses["loss_aux"].detach()), batch_size)

        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            final=f"{final_loss_meter.avg:.4f}",
            aux=f"{aux_loss_meter.avg:.4f}",
        )

    return {
        "train_loss": loss_meter.avg,
        "train_loss_final": final_loss_meter.avg,
        "train_loss_aux": aux_loss_meter.avg,
    }

def find_best_threshold(y_true, y_prob, mode="youden", target_sensitivity=0.90):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if mode == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j = tpr - fpr
        idx = np.argmax(j)
        return float(thresholds[idx])

    if mode == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        idx = np.argmax(f1)
        return float(thresholds[idx])

    if mode == "sensitivity":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        valid = np.where(tpr >= target_sensitivity)[0]
        if len(valid) == 0:
            return 0.5
        idx = valid[np.argmin(fpr[valid])]
        return float(thresholds[idx])

    return 0.5

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    final_pos_weight: Optional[torch.Tensor] = None,
    aux_pos_weight: Optional[torch.Tensor] = None,
    final_threshold=0.5
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    model.eval()

    has_targets = True
    if hasattr(loader.dataset, "require_targets"):
        has_targets = bool(getattr(loader.dataset, "require_targets"))

    loss_meter = AverageMeter()
    final_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    final_targets: List[float] = []
    final_probs: List[float] = []
    aux_targets: List[List[float]] = [[] for _ in AUX_COLUMNS]
    aux_probs: List[List[float]] = [[] for _ in AUX_COLUMNS]
    pred_rows: List[Dict[str, object]] = []

    progress = tqdm(loader, desc=f"Valid {epoch}", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(
                batch["image"],
                geometry=batch.get("geometry"),
                geometry_mask=batch.get("geometry_mask"),
            )
            if has_targets:
                losses = compute_loss(
                    outputs,
                    batch,
                    final_pos_weight=final_pos_weight,
                    aux_pos_weight=aux_pos_weight,
                )

        batch_size = batch["image"].size(0)
        batch_probs_final = torch.sigmoid(outputs["final_logits"]).detach().cpu().numpy()
        batch_probs_aux = torch.sigmoid(outputs["aux_logits"]).detach().cpu().numpy()

        if has_targets:
            loss_meter.update(float(losses["loss"].detach()), batch_size)
            final_loss_meter.update(float(losses["loss_final"].detach()), batch_size)
            aux_loss_meter.update(float(losses["loss_aux"].detach()), batch_size)

            final_targets.extend(batch["final"].detach().cpu().numpy().tolist())
            final_probs.extend(batch_probs_final.tolist())

            aux_batch_targets = batch["aux"].detach().cpu().numpy()
            aux_batch_masks = batch["aux_mask"].detach().cpu().numpy()
            for task_idx in range(len(AUX_COLUMNS)):
                valid = aux_batch_masks[:, task_idx] > 0
                if valid.any():
                    aux_targets[task_idx].extend(aux_batch_targets[valid, task_idx].tolist())
                    aux_probs[task_idx].extend(batch_probs_aux[valid, task_idx].tolist())

        image_ids = batch["image_id"]
        if isinstance(image_ids, (list, tuple)):
            image_ids_iter = image_ids
        else:
            image_ids_iter = [image_ids]

        for idx, image_id in enumerate(image_ids_iter):
            row: Dict[str, object] = {
                "Image": image_id,
                "epoch": int(epoch),
                "final_prob": float(batch_probs_final[idx]),
                "final_pred": int(batch_probs_final[idx] >= final_threshold),
            }
            for task_idx, task_name in enumerate(AUX_COLUMNS):
                row[f"{task_name}_prob"] = float(batch_probs_aux[idx, task_idx])
                row[f"{task_name}_pred"] = int(batch_probs_aux[idx, task_idx] >= 0.5)
            if has_targets:
                row["final_target"] = float(batch["final"][idx].detach().cpu().item())
                for task_idx, task_name in enumerate(AUX_COLUMNS):
                    row[f"{task_name}_target"] = float(batch["aux"][idx, task_idx].detach().cpu().item())
                    row[f"{task_name}_mask"] = float(batch["aux_mask"][idx, task_idx].detach().cpu().item())
            pred_rows.append(row)

        if has_targets:
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")
        else:
            progress.set_postfix(samples=len(pred_rows))

    metrics: Dict[str, float] = {}
    if has_targets:
        metrics = {
            "val_loss": loss_meter.avg,
            "val_loss_final": final_loss_meter.avg,
            "val_loss_aux": aux_loss_meter.avg,
            "val_auroc_final": safe_auroc(final_targets, final_probs),
        }

        aux_aurocs = []
        for name, targets, probs in zip(AUX_COLUMNS, aux_targets, aux_probs):
            auc = safe_auroc(targets, probs)
            metrics[f"val_auroc_{name}"] = auc
            if not math.isnan(auc):
                aux_aurocs.append(auc)

        metrics["val_auroc_aux_mean"] = float(np.mean(aux_aurocs)) if aux_aurocs else float("nan")
    else:
        metrics = {"eval_samples": float(len(pred_rows))}

    return metrics, pd.DataFrame(pred_rows) if pred_rows else None


def save_checkpoint(
    output_dir: Path,
    filename: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    metrics: Dict[str, float],
    best_val_auroc: float,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "best_val_auroc": best_val_auroc,
        "args": vars(args),
        "aux_columns": AUX_COLUMNS,
    }
    torch.save(checkpoint, output_dir / filename)


def warm_start_from_stage_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> Dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    checkpoint_state = checkpoint["model_state_dict"]
    current_state = model.state_dict()

    copied_matching = 0
    skipped_mismatch = 0
    aux_mapped = 0

    aux_weight_key = "aux_head.2.weight"
    aux_bias_key = "aux_head.2.bias"

    filtered_state: Dict[str, torch.Tensor] = {}
    for key, tensor in checkpoint_state.items():
        if key in {aux_weight_key, aux_bias_key}:
            continue
        if key in current_state and current_state[key].shape == tensor.shape:
            filtered_state[key] = tensor
            copied_matching += 1
        else:
            skipped_mismatch += 1

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)

    source_aux_columns = checkpoint.get("aux_columns", []) or []
    target_aux_columns = list(AUX_COLUMNS)
    source_aux_index = {name: idx for idx, name in enumerate(source_aux_columns)}

    target_aux_linear = model.aux_head[-1]
    if not isinstance(target_aux_linear, nn.Linear):
        raise TypeError("Expected aux_head to end with nn.Linear.")

    source_aux_weight = checkpoint_state.get(aux_weight_key)
    source_aux_bias = checkpoint_state.get(aux_bias_key)

    if source_aux_weight is not None and source_aux_bias is not None:
        with torch.no_grad():
            for target_idx, task_name in enumerate(target_aux_columns):
                source_idx = source_aux_index.get(task_name)
                if source_idx is None:
                    continue
                if source_idx >= source_aux_weight.shape[0]:
                    continue
                target_aux_linear.weight[target_idx].copy_(source_aux_weight[source_idx].to(target_aux_linear.weight.device))
                target_aux_linear.bias[target_idx].copy_(source_aux_bias[source_idx].to(target_aux_linear.bias.device))
                aux_mapped += 1

    print(
        f"Warm-started from {checkpoint_path} | copied={copied_matching} | "
        f"aux_mapped={aux_mapped}/{len(target_aux_columns)} | "
        f"missing_keys={len(missing_keys)} | unexpected_keys={len(unexpected_keys)} | "
        f"shape_skipped={skipped_mismatch}"
    )
    return {
        "copied_matching": copied_matching,
        "aux_mapped": aux_mapped,
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
        "shape_skipped": skipped_mismatch,
    }


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_auroc = float(checkpoint.get("best_val_auroc", float("-inf")))
    return start_epoch, best_val_auroc


def resolve_warm_start_checkpoint(args: argparse.Namespace) -> Optional[Path]:
    if args.warm_start_checkpoint:
        return Path(args.warm_start_checkpoint)

    default_output_root = Path(DEFAULT_OUTPUT_DIR)
    if args.stage == "stage2":
        candidate = default_output_root / "stage_runs_stage1" / "best.pt"
        return candidate if candidate.exists() else None
    if args.stage == "stage3":
        stage2_candidate = default_output_root / "stage_runs_stage2" / "best.pt"
        if stage2_candidate.exists():
            return stage2_candidate
        stage1_candidate = default_output_root / "stage_runs_stage1" / "best.pt"
        return stage1_candidate if stage1_candidate.exists() else None
    return None


def save_predictions_csv(predictions: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)


def save_confusion_matrix_figure(
    predictions: pd.DataFrame,
    output_path: Path,
    title: str,
    target_col: str = "final_target",
    pred_col: str = "final_pred",
    mask_col: Optional[str] = None,
    label_names: Optional[List[str]] = None,
) -> Optional[Path]:
    if predictions is None or predictions.empty:
        return None
    if target_col not in predictions.columns or pred_col not in predictions.columns:
        return None

    frame = predictions
    if mask_col is not None:
        if mask_col not in frame.columns:
            return None
        frame = frame[frame[mask_col].astype(float) > 0]
        if frame.empty:
            return None

    y_true = frame[target_col].astype(int).to_numpy()
    y_pred = frame[pred_col].astype(int).to_numpy()
    labels = [0, 1]
    label_names = label_names or ["0", "1"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=160)
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = float(normalized[i, j] * 100.0)
            ax.text(
                j,
                i,
                f"{count}\n{pct:.1f}%",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=11,
            )

    ax.set_ylim(len(label_names) - 0.5, -0.5)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _valid_prediction_frame(
    predictions: pd.DataFrame,
    target_col: str,
    pred_col: str,
    mask_col: Optional[str],
) -> pd.DataFrame:
    if target_col not in predictions.columns or pred_col not in predictions.columns:
        return pd.DataFrame()

    frame = predictions[predictions[target_col].notna() & predictions[pred_col].notna()]
    if mask_col is not None:
        if mask_col not in frame.columns:
            return pd.DataFrame()
        frame = frame[frame[mask_col].astype(float) > 0]
    return frame


def _format_correct_over_valid(correct: int, valid: int) -> str:
    if valid <= 0:
        return "0/0 (-)"
    return f"{correct}/{valid} ({correct / valid * 100.0:.1f}%)"


def save_multitask_confusion_matrices(
    predictions: pd.DataFrame,
    output_dir: Path,
    split_name: str,
    epoch: Optional[int] = None,
) -> Optional[Path]:
    if predictions is None or predictions.empty:
        return None

    task_specs = [
        {
            "task_name": "final",
            "title": "Final",
            "target_col": "final_target",
            "pred_col": "final_pred",
            "mask_col": None,
            "label_names": ["NRG", "RG"],
        }
    ]
    for aux_name in AUX_COLUMNS:
        task_specs.append(
            {
                "task_name": aux_name,
                "title": aux_name,
                "target_col": f"{aux_name}_target",
                "pred_col": f"{aux_name}_pred",
                "mask_col": f"{aux_name}_mask",
                "label_names": ["0", "1"],
            }
        )

    epoch_suffix = f"_epoch_{epoch:03d}" if epoch is not None and epoch > 0 else ""
    output_path = output_dir / "confusion_matrices" / f"{split_name}{epoch_suffix}_all_tasks.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_cols = 4
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 14), dpi=160)
    axes_flat = axes.ravel()

    figure_title = f"{split_name.capitalize()} Confusion Matrices"
    if epoch is not None and epoch > 0:
        figure_title += f" - Epoch {epoch}"
    fig.suptitle(figure_title, fontsize=18, y=0.995)

    labels = [0, 1]
    plotted = 0
    for axis, spec in zip(axes_flat, task_specs):
        target_col = spec["target_col"]
        pred_col = spec["pred_col"]
        mask_col = spec["mask_col"]
        label_names = spec["label_names"]

        frame = _valid_prediction_frame(predictions, target_col, pred_col, mask_col)
        if frame.empty:
            axis.set_title(f"{spec['title']}\nNo valid labels", fontsize=12)
            axis.axis("off")
            continue

        y_true = frame[target_col].astype(int).to_numpy()
        y_pred = frame[pred_col].astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        valid_0 = int(cm[0, :].sum())
        valid_1 = int(cm[1, :].sum())
        correct_0 = int(cm[0, 0])
        correct_1 = int(cm[1, 1])

        axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set(
            xticks=np.arange(len(label_names)),
            yticks=np.arange(len(label_names)),
            xticklabels=label_names,
            yticklabels=label_names,
            xlabel="Pred",
            ylabel="True",
            title=(
                f"{spec['title']}\n"
                f"0: {_format_correct_over_valid(correct_0, valid_0)} | "
                f"1: {_format_correct_over_valid(correct_1, valid_1)}"
            ),
        )
        axis.tick_params(axis="both", labelsize=9)

        threshold = 0.5
        for row_idx in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                count = int(cm[row_idx, col_idx])
                pct = float(normalized[row_idx, col_idx] * 100.0)
                axis.text(
                    col_idx,
                    row_idx,
                    f"{count}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if normalized[row_idx, col_idx] > threshold else "black",
                    fontsize=10,
                )
        axis.set_ylim(len(label_names) - 0.5, -0.5)
        plotted += 1

    for axis in axes_flat[len(task_specs):]:
        axis.axis("off")

    if plotted == 0:
        plt.close(fig)
        return None

    fig.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.06, wspace=0.34, hspace=0.58)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path if plotted > 0 else None


def denormalize_image_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor((0.485, 0.456, 0.406), device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=image_tensor.device).view(3, 1, 1)
    image = image_tensor.detach() * std + mean
    image = image.clamp(0.0, 1.0)
    image_np = (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return image_np


def estimate_fundus_mask(image_rgb: np.ndarray, threshold: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > threshold).astype(np.uint8)
    mask = cv2.medianBlur(mask * 255, 5)
    return mask > 0


def normalize_cam_within_mask(cam_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cam = cam_map.astype(np.float32, copy=True)
    if mask.any():
        valid_values = cam[mask]
        cam_min = float(valid_values.min())
        cam_max = float(valid_values.max())
    else:
        cam_min = float(cam.min())
        cam_max = float(cam.max())
    cam = cam - cam_min
    cam = cam / max(cam_max - cam_min, 1e-8)
    if mask.any():
        cam[~mask] = 0.0
    return np.clip(cam, 0.0, 1.0)


def render_gradcam_overlay(image_rgb: np.ndarray, cam_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    fundus_mask = estimate_fundus_mask(image_rgb)
    cam_map = normalize_cam_within_mask(cam_map, fundus_mask)
    cam_map = cv2.GaussianBlur(cam_map, ksize=(0, 0), sigmaX=2.0, sigmaY=2.0)
    cam_map = normalize_cam_within_mask(cam_map, fundus_mask)
    cam_uint8 = np.clip(cam_map * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0.0)
    if fundus_mask.any():
        overlay[~fundus_mask] = image_rgb[~fundus_mask]
    return overlay


import cv2
import numpy as np


def annotate_panel(panel: np.ndarray, title: str) -> np.ndarray:
    annotated = panel.copy()
    h, w = annotated.shape[:2]

    # 폰트 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4  # 기본 크기를 조금 줄였습니다
    thickness = 1

    # 텍스트 크기 측정
    (text_w, text_h), baseline = cv2.getTextSize(title, font, font_scale, thickness)

    # 만약 텍스트가 패널 너비보다 크면, 비율에 맞춰 폰트 크기 축소
    if text_w > w - 10:
        font_scale = (w - 20) / text_w * font_scale
        (text_w, text_h), baseline = cv2.getTextSize(title, font, font_scale, thickness)

    # 텍스트 위치 (이미지 상단 여백 확보)
    org = (8, text_h + 8)

    # 검정색 테두리 효과 (가독성 향상)
    cv2.putText(annotated, title, org, font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # 흰색 텍스트
    cv2.putText(annotated, title, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated


def compute_single_task_gradcam(
    model: JustRAIGSConvNeXtMoE,
    image: torch.Tensor,
    geometry: Optional[torch.Tensor],
    geometry_mask: Optional[torch.Tensor],
    task_index: int,
    is_final_task: bool,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)

    spatial_features = model._backbone_forward_spatial(image)
    spatial_features.retain_grad()
    pooled_features = model._pool_spatial_features(spatial_features)
    outputs = model._forward_heads_from_pooled(
        pooled_features,
        geometry=geometry,
        geometry_mask=geometry_mask,
    )
    target_logit = outputs["final_logits"][0] if is_final_task else outputs["aux_logits"][0, task_index]
    target_logit.backward()

    gradients = spatial_features.grad
    if gradients is None:
        raise RuntimeError("Gradients were not retained for Grad-CAM computation.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * spatial_features).sum(dim=1, keepdim=False)
    cam = torch.relu(cam)
    cam = cam[0].detach().cpu().numpy()
    cam = cv2.resize(cam, dsize=(image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_LINEAR)
    cam = cam - cam.min()
    cam = cam / max(cam.max(), 1e-8)
    return cam


def save_gradcam_visualizations(
    model: JustRAIGSConvNeXtMoE,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    split_name: str,
    max_samples: int,
) -> None:
    if max_samples <= 0:
        return

    gradcam_dir = output_dir / "gradcam" / split_name
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    print(f"Grad-CAM output dir -> {gradcam_dir}")


    model.eval()
    saved_by_bucket = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "unlabeled": 0}
    per_bucket_quota = max(1, math.ceil(max_samples / 4))
    label_map = {0: "N", 1: "T"}
    task_names = [("final", True, -1)] + [(task_name, False, task_idx) for task_idx, task_name in enumerate(AUX_COLUMNS)]

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        image_ids = batch["image_id"]
        if not isinstance(image_ids, (list, tuple)):
            image_ids = [image_ids]

        for batch_index, image_id in enumerate(image_ids):
            labeled_bucket_total = sum(saved_by_bucket[name] for name in ["tp", "tn", "fp", "fn"])
            if labeled_bucket_total >= max_samples:
                print(f"Saved Grad-CAM visualizations -> {gradcam_dir}")
                print(f"Grad-CAM bucket counts -> {saved_by_bucket}")
                return

            image_tensor = batch["image"][batch_index : batch_index + 1]
            geometry = batch.get("geometry")
            geometry_mask = batch.get("geometry_mask")
            geometry_sample = geometry[batch_index : batch_index + 1] if torch.is_tensor(geometry) else None
            geometry_mask_sample = geometry_mask[batch_index : batch_index + 1] if torch.is_tensor(geometry_mask) else None
            image_rgb = denormalize_image_tensor(image_tensor[0])
            with torch.no_grad():
                outputs = model(
                    image_tensor,
                    geometry=geometry_sample,
                    geometry_mask=geometry_mask_sample,
                )
                final_prob = float(torch.sigmoid(outputs["final_logits"])[0].detach().cpu().item())
                aux_probs = torch.sigmoid(outputs["aux_logits"])[0].detach().cpu().numpy()

            final_pred = int(final_prob >= 0.5)
            final_target = None
            if "final" in batch and torch.is_tensor(batch["final"]):
                final_target = int(float(batch["final"][batch_index].detach().cpu().item()) >= 0.5)

            if final_target is None:
                outcome_bucket = "unlabeled"
            elif final_target == 1 and final_pred == 1:
                outcome_bucket = "tp"
            elif final_target == 0 and final_pred == 0:
                outcome_bucket = "tn"
            elif final_target == 0 and final_pred == 1:
                outcome_bucket = "fp"
            else:
                outcome_bucket = "fn"

            if outcome_bucket != "unlabeled":
                if saved_by_bucket[outcome_bucket] >= per_bucket_quota:
                    continue
            else:
                if saved_by_bucket[outcome_bucket] >= max(1, min(2, max_samples)):
                    continue

            sample_dir = gradcam_dir / outcome_bucket
            sample_dir.mkdir(parents=True, exist_ok=True)
            pred_letter = label_map.get(final_pred, final_pred)
            base_title = f"orig final p={final_prob:.2f} pred={pred_letter}"
            if final_target is not None:
                tgt_letter = label_map.get(final_target, final_target)
                base_title += f" tgt={tgt_letter}"
            task_panels: List[np.ndarray] = [annotate_panel(image_rgb, base_title)]

            for task_name, is_final_task, task_index in task_names:
                model.zero_grad(set_to_none=True)
                cam_map = compute_single_task_gradcam(
                    model=model,
                    image=image_tensor,
                    geometry=geometry_sample,
                    geometry_mask=geometry_mask_sample,
                    task_index=task_index,
                    is_final_task=is_final_task,
                )
                overlay = render_gradcam_overlay(image_rgb, cam_map)
                if is_final_task:
                    task_prob = final_prob
                    task_pred = final_pred
                    task_target = final_target
                else:
                    task_prob = float(aux_probs[task_index])
                    task_pred = int(task_prob >= 0.5)
                    if "aux" in batch and torch.is_tensor(batch["aux"]):
                        task_target = int(float(batch["aux"][batch_index, task_index].detach().cpu().item()) >= 0.5)
                    else:
                        task_target = None

                task_pred_letter = label_map.get(task_pred, task_pred)
                title = f"{task_name} p={task_prob:.2f} pred={task_pred_letter}"
                if task_target is not None:
                    task_tgt_letter = label_map.get(task_target, task_target)
                    title += f" tgt={task_tgt_letter}"
                # title = f"{task_name} p={task_prob:.2f} pred={task_pred}"
                # if task_target is not None:
                #     title += f" tgt={task_target}"
                panel = annotate_panel(overlay, title)
                task_panels.append(panel)

                # task_output_path = sample_dir / f"{image_id}_{task_name}.jpg"
                # cv2.imwrite(str(task_output_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            if task_panels:
                n_cols = 3
                panel_h, panel_w = task_panels[0].shape[:2]
                n_rows = math.ceil(len(task_panels) / n_cols)
                grid = np.full((n_rows * panel_h, n_cols * panel_w, 3), 24, dtype=np.uint8)
                for panel_idx, panel in enumerate(task_panels):
                    row_idx = panel_idx // n_cols
                    col_idx = panel_idx % n_cols
                    y0 = row_idx * panel_h
                    x0 = col_idx * panel_w
                    grid[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel
                grid_output_path = sample_dir / f"{image_id}_grid.jpg"
                grid_saved = cv2.imwrite(str(grid_output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                if not grid_saved:
                    print(f"Warning: failed to save Grad-CAM grid -> {grid_output_path}")
                elif sum(saved_by_bucket.values()) < 5:
                    print(f"Saved Grad-CAM grid -> {grid_output_path}")

            saved_by_bucket[outcome_bucket] += 1

    print(f"Saved Grad-CAM visualizations -> {gradcam_dir}")
    print(f"Grad-CAM bucket counts -> {saved_by_bucket}")
    total_saved = sum(saved_by_bucket.values())
    if total_saved == 0:
        print(
            "Warning: no Grad-CAM samples were saved. "
            "Check checkpoint loading, split selection, and whether the selected loader produced samples."
        )


def format_metrics(metrics: Dict[str, float]) -> str:
    preferred = [
        "epoch",
        "eval_samples",
        "eval_loss",
        "eval_loss_final",
        "eval_loss_aux",
        "eval_auroc_final",
        "eval_auroc_aux_mean",
        "train_loss",
        "train_loss_final",
        "train_loss_aux",
        "val_loss",
        "val_loss_final",
        "val_loss_aux",
        "val_auroc_final",
        "val_auroc_aux_mean",
        "test_loss",
        "test_loss_final",
        "test_loss_aux",
        "test_auroc_final",
        "test_auroc_aux_mean",
        "lr_backbone",
        "lr_head",
        "val_prob_mean_final",
        "val_prob_std_final",
        "val_pred_pos_rate_final",
    ]
    parts = []
    for key in preferred:
        if key in metrics:
            value = metrics[key]
            parts.append(f"{key}={value:.4f}" if not math.isnan(value) else f"{key}=nan")
    return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-task JustRAIGS glaucoma training pipeline.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Training CSV or full CSV.")
    parser.add_argument("--val-csv", type=str, default=None, help="Optional validation CSV.")
    parser.add_argument("--eval-csv", "--test-csv", dest="eval_csv", type=str, default=None, help="CSV to evaluate/test.")
    parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR, help="Image or .npy cache root.")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Directory with images.npy and paths.npy.")
    parser.add_argument("--cache-images-name", type=str, default="images.npy")
    parser.add_argument("--cache-paths-name", type=str, default="paths.npy")
    parser.add_argument(
        "--cache-lookup",
        type=str,
        default="path",
        choices=["path", "index"],
        help="Use paths.npy matching or CSV row index lookup for images.npy.",
    )
    parser.add_argument("--allow-unlabeled-eval", action="store_true", help="Allow eval CSV without target columns and only export predictions.")
    parser.add_argument("--disable-npy-cache", action="store_true", help="Read individual image files instead of images.npy.")
    parser.add_argument("--disable-cache-mmap", action="store_true", help="Load the full images.npy into RAM.")
    parser.add_argument(
        "--disable-cache-index-fallback",
        action="store_true",
        help="Do not fall back to CSV row index when Image cannot be matched to paths.npy.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Checkpoint directory.")
    parser.add_argument("--checkpoint", type=str, default=str(Path(DEFAULT_OUTPUT_DIR) / "best.pt"), help="Checkpoint path to load for eval-only or warm start.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run checkpoint evaluation only.")
    parser.add_argument("--preds-csv", type=str, default=None, help="Optional path to save evaluation predictions.")
    parser.add_argument("--metrics-csv", type=str, default=None, help="Optional path to save per-epoch metrics as CSV.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument(
        "--warm-start-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to initialize stage2/stage3 from without restoring optimizer/scheduler state.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage1",
        choices=list(STAGE_TASKS.keys()),
        help="Stage-wise auxiliary-task preset. stage1 keeps denser auxiliaries, stage2 adds rarer ones, stage3 uses all.",
    )
    parser.add_argument(
        "--aux-tasks",
        type=str,
        default=None,
        help="Optional comma-separated auxiliary tasks override, e.g. ANRS,ANRI,LD,LC,NVT",
    )
    parser.add_argument(
        "--run-stage2-after-stage1",
        action="store_true",
        help="After finishing stage1, automatically launch stage2 training in the same script.",
    )

    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="timm backbone model name, e.g. swin_tiny.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use timm pretrained weights if available.")
    parser.add_argument("--disable-pretrained", dest="pretrained", action="store_false", help="Disable timm pretrained weights.")
    parser.add_argument(
        "--head-type",
        type=str,
        default="simple",
        choices=["simple", "moe"],
        help="Classification head type. simple is more stable; moe keeps the previous mixture-of-experts head.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop training if val_auroc_final does not improve for this many consecutive epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val_auroc_final improvement required to reset early stopping patience.",
    )
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=None, help="Deprecated single learning rate. If set, overrides both backbone/head lr.")
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--moe-dim", type=int, default=512)
    parser.add_argument("--geometry-csv", type=str, default=None, help="Optional CSV with segmentation-derived geometry features keyed by Image.")
    parser.add_argument(
        "--geometry-features",
        type=str,
        default=",".join(GEOMETRY_FEATURE_COLUMNS),
        help="Comma-separated geometry feature columns to use from --geometry-csv.",
    )
    parser.add_argument("--geometry-hidden-dim", type=int, default=128)
    parser.add_argument("--geometry-normalize", dest="geometry_normalize", action="store_true", help="Normalize geometry features using train-split mean/std.")
    parser.add_argument("--disable-geometry-normalize", dest="geometry_normalize", action="store_false", help="Disable train-split normalization for geometry features.")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-gradcam", dest="save_gradcam", action="store_true", help="Save Grad-CAM overlays for final and auxiliary tasks.")
    parser.add_argument("--disable-save-gradcam", dest="save_gradcam", action="store_false", help="Disable Grad-CAM export.")
    parser.add_argument("--gradcam-samples", type=int, default=20, help="Number of samples to export Grad-CAM overlays for.")
    parser.add_argument(
        "--gradcam-split",
        type=str,
        default="test",
        choices=["val", "test", "eval"],
        help="Which split to render Grad-CAM visualizations from.",
    )
    parser.add_argument(
        "--final-pos-weight-max",
        type=float,
        default=10.0,
        help="Upper cap for final-label BCE positive weight computed from the train split.",
    )
    parser.add_argument(
        "--aux-pos-weight-max",
        type=float,
        default=10.0,
        help="Upper cap for per-auxiliary-task BCE positive weights computed from the train split.",
    )
    parser.add_argument(
        "--pos-weight-power",
        type=float,
        default=0.5,
        help="Exponent applied to neg/pos ratio for BCE pos_weight. 0.5 means sqrt ratio.",
    )
    parser.add_argument(
        "--balanced-final-sampler",
        action="store_true",
        help="Use weighted sampling to balance Final=0 and Final=1 in training batches.",
    )
    parser.add_argument(
        "--disable-balanced-final-sampler",
        action="store_true",
        help="Compatibility flag; keeps weighted final-label sampling disabled.",
    )
    parser.add_argument(
        "--disable-prior-bias-init",
        action="store_true",
        help="Disable initializing output head biases from train-split positive priors.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.set_defaults(pretrained=True, geometry_normalize=True, save_gradcam=True)
    args = parser.parse_args()
    if args.lr is not None:
        args.backbone_lr = args.lr
        args.head_lr = args.lr
    return args


def run_single_stage(args: argparse.Namespace) -> None:
    selected_aux_tasks = resolve_selected_aux_tasks(args)
    configure_aux_tasks(selected_aux_tasks)
    args.geometry_feature_columns = [
        column.strip() for column in args.geometry_features.split(",") if column.strip()
    ] if args.geometry_csv else []
    args.geometry_norm_stats = {}
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    output_dir = _stage_output_dir(args.output_dir, args.stage)

    model = JustRAIGSConvNeXtMoE(
        model_name=args.model_name,
        pretrained=args.pretrained,
        aux_tasks=len(AUX_COLUMNS),
        num_experts=args.num_experts,
        moe_dim=args.moe_dim,
        dropout=args.dropout,
        head_type=args.head_type,
        geometry_dim=len(args.geometry_feature_columns),
        geometry_hidden_dim=args.geometry_hidden_dim,
    ).to(device)

    optimizer = build_optimizer(
        model=model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if args.eval_only:
        eval_loader = build_eval_loader(args)
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            device=device,
        )

        metrics, predictions = validate(
            model=model,
            loader=eval_loader,
            device=device,
            epoch=0,
            amp_enabled=amp_enabled,
        )
        metrics = rename_metric_prefix(metrics, "val_", "eval_")

        print(f"Eval checkpoint: {checkpoint_path}")
        print(format_metrics(metrics))
        if predictions is not None:
            saved_cm_path = save_multitask_confusion_matrices(
                predictions=predictions,
                output_dir=output_dir,
                split_name="eval",
                epoch=None,
            )
            if saved_cm_path is not None:
                print(f"Saved multitask confusion matrix to {saved_cm_path}")
        if predictions is not None and args.preds_csv:
            save_predictions_csv(predictions, Path(args.preds_csv))
            print(f"Saved predictions to {args.preds_csv}")
        if args.save_gradcam:
            if args.gradcam_split != "eval":
                print(
                    f"Grad-CAM note -> eval-only mode uses eval_loader, "
                    f"so requested gradcam_split='{args.gradcam_split}' is being redirected to 'eval'."
                )
            save_gradcam_visualizations(
                model=model,
                loader=eval_loader,
                device=device,
                output_dir=output_dir,
                split_name="eval",
                max_samples=args.gradcam_samples,
            )
        return

    (
        train_loader,
        val_loader,
        test_loader,
        final_pos_weight,
        aux_pos_weight,
        final_prior,
        aux_priors,
    ) = build_loaders(args)
    final_pos_weight = final_pos_weight.to(device)
    aux_pos_weight = aux_pos_weight.to(device)

    if not args.resume and not args.disable_prior_bias_init:
        initialize_output_biases(model, final_prior=final_prior, aux_priors=aux_priors)
        print("Initialized output head biases from train-split positive priors.")

    warm_start_checkpoint = resolve_warm_start_checkpoint(args)
    if args.resume is None and warm_start_checkpoint is not None:
        if warm_start_checkpoint.exists():
            warm_start_from_stage_checkpoint(
                checkpoint_path=warm_start_checkpoint,
                model=model,
                device=device,
            )
        else:
            print(f"Warm-start checkpoint not found, skipping: {warm_start_checkpoint}")

    print(
        "Training setup -> "
        f"model={resolve_model_name(args.model_name)} | pretrained={args.pretrained} | "
        f"head_type={args.head_type} | backbone_lr={args.backbone_lr:.2e} | "
        f"head_lr={args.head_lr:.2e} | batch_size={args.batch_size} | image_size={args.image_size} | "
        f"stage={args.stage} | aux_tasks={AUX_COLUMNS if AUX_COLUMNS else ['final_only']} | "
        f"geometry={'on' if args.geometry_feature_columns else 'off'} | "
        f"geometry_dim={len(args.geometry_feature_columns)} | "
        f"geometry_normalize={args.geometry_normalize} | "
        f"warm_start={str(warm_start_checkpoint) if warm_start_checkpoint is not None else 'none'}"
    )

    start_epoch = 1
    best_val_auroc = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    metrics_history: List[Dict[str, float]] = []
    if args.resume:
        start_epoch, best_val_auroc = load_checkpoint(
            Path(args.resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            amp_enabled=amp_enabled,
            grad_clip_norm=args.grad_clip_norm,
            final_pos_weight=final_pos_weight,
            aux_pos_weight=aux_pos_weight,
        )
        val_metrics, val_predictions = validate(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            amp_enabled=amp_enabled,
            final_pos_weight=final_pos_weight,
            aux_pos_weight=aux_pos_weight,
        )
        best_threshold = 0.5
        if val_predictions is not None:
            best_threshold = find_best_threshold(
                val_predictions["final_target"],
                val_predictions["final_prob"],
                mode="youden",
            )
            val_predictions["final_pred"] = (
                    val_predictions["final_prob"].astype(float) >= best_threshold
            ).astype(int)
            print(f"Best validation threshold by Youden J = {best_threshold:.4f}")
        if val_predictions is not None:
            saved_cm_path = save_multitask_confusion_matrices(
                predictions=val_predictions,
                output_dir=output_dir,
                split_name="val",
                epoch=epoch,
            )
            if saved_cm_path is not None:
                print(f"Saved validation multitask confusion matrix to {saved_cm_path}")

        scheduler.step()

        metrics = {
            **train_metrics,
            **val_metrics,
            **collect_probability_summary(val_predictions),
            "epoch": float(epoch),
            "lr_backbone": _as_float(optimizer.param_groups[0]["lr"]),
            "lr_head": _as_float(optimizer.param_groups[1]["lr"]),
        }
        metrics_history.append(metrics.copy())
        metrics_csv_path = Path(args.metrics_csv) if args.metrics_csv else output_dir / "metrics_history.csv"
        save_metrics_history(metrics_history, metrics_csv_path)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | {format_metrics(metrics)}")
        if "val_prob_mean_final" in metrics:
            print(
                "Validation probability summary -> "
                f"Final mean={metrics['val_prob_mean_final']:.4f}, "
                f"std={metrics['val_prob_std_final']:.4f}, "
                f"pred_pos_rate={metrics['val_pred_pos_rate_final']:.4f}"
            )

        current_auroc = metrics["val_auroc_final"]
        improved = not math.isnan(current_auroc) and current_auroc > (best_val_auroc + args.early_stopping_min_delta)
        is_best = improved
        if is_best:
            best_val_auroc = current_auroc
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir=output_dir,
                filename="best.pt",
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
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0:
                print(
                    f"No validation AUROC improvement for {epochs_without_improvement} epoch(s). "
                    f"Best epoch={best_epoch:03d}, best_val_auroc_final={best_val_auroc:.4f}"
                )

        save_checkpoint(
            output_dir=output_dir,
            filename="last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            metrics=metrics,
            best_val_auroc=best_val_auroc,
            args=args,
        )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch:03d}. "
                f"Best epoch={best_epoch:03d}, best_val_auroc_final={best_val_auroc:.4f}"
            )
            break

    best_checkpoint_path = output_dir / "best.pt"
    final_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else output_dir / "last.pt"
    if final_checkpoint_path.exists():
        load_checkpoint(
            final_checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            device=device,
        )
        _, val_predictions_for_thr = validate(
            model=model,
            loader=val_loader,
            device=device,
            epoch=0,
            amp_enabled=amp_enabled,
            final_pos_weight=final_pos_weight,
            aux_pos_weight=aux_pos_weight,
        )

        best_threshold = find_best_threshold(
            val_predictions_for_thr["final_target"],
            val_predictions_for_thr["final_prob"],
            mode="youden",
        )
        print(f"Fixed threshold from validation set = {best_threshold:.4f}")
        test_metrics, test_predictions = validate(
            model=model,
            loader=test_loader,
            device=device,
            epoch=0,
            amp_enabled=amp_enabled,
            final_pos_weight=final_pos_weight,
            aux_pos_weight=aux_pos_weight,
            final_threshold=best_threshold
        )

        test_metrics = rename_metric_prefix(test_metrics, "val_", "test_")
        print(f"Test checkpoint: {final_checkpoint_path}")
        print(format_metrics(test_metrics))
        if test_predictions is not None:
            saved_cm_path = save_multitask_confusion_matrices(
                predictions=test_predictions,
                output_dir=output_dir,
                split_name="test",
                epoch=None,
            )
            if saved_cm_path is not None:
                print(f"Saved test multitask confusion matrix to {saved_cm_path}")
        if test_predictions is not None and args.preds_csv:
            save_predictions_csv(test_predictions, Path(args.preds_csv))
            print(f"Saved predictions to {args.preds_csv}")
        if args.save_gradcam:
            gradcam_loader = test_loader if args.gradcam_split == "test" else val_loader
            save_gradcam_visualizations(
                model=model,
                loader=gradcam_loader,
                device=device,
                output_dir=output_dir,
                split_name=args.gradcam_split,
                max_samples=args.gradcam_samples,
            )


def main() -> None:
    args = parse_args()
    if args.eval_only:
        checkpoint_aux_columns = read_checkpoint_aux_columns(Path(args.checkpoint))
        if checkpoint_aux_columns:
            args.aux_tasks = ",".join(checkpoint_aux_columns)
            print(
                "Eval-only checkpoint aux task override -> "
                f"{checkpoint_aux_columns}"
            )
    seed_everything(args.seed)
    run_single_stage(args)

    if args.run_stage2_after_stage1 and args.stage == "stage1":
        stage2_args = copy.deepcopy(args)
        stage2_args.stage = "stage2"
        stage2_args.aux_tasks = None
        stage2_args.resume = None
        stage2_args.eval_only = False
        stage2_args.checkpoint = str(_stage_output_dir(args.output_dir, "stage2") / "best.pt")
        stage2_args.warm_start_checkpoint = str(_stage_output_dir(args.output_dir, "stage1") / "best.pt")
        stage2_args.geometry_norm_stats = {}
        if args.metrics_csv:
            base_metrics_path = Path(args.metrics_csv)
            stage2_args.metrics_csv = str(base_metrics_path.with_name(f"{base_metrics_path.stem}_stage2{base_metrics_path.suffix}"))
        if args.preds_csv:
            base_preds_path = Path(args.preds_csv)
            stage2_args.preds_csv = str(base_preds_path.with_name(f"{base_preds_path.stem}_stage2{base_preds_path.suffix}"))

        print("\n" + "=" * 80)
        print("Stage1 complete. Starting automatic Stage2 training...")
        print(f"Stage2 warm start checkpoint -> {stage2_args.warm_start_checkpoint}")
        print("=" * 80)
        run_single_stage(stage2_args)


if __name__ == "__main__":
    main()
