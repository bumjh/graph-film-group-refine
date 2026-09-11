import argparse
import copy
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    from . import train_justraigs_multitask_stagewise as base
except ImportError:
    import train_justraigs_multitask_stagewise as base


DEFAULT_RGS_V3_DIR = str(Path(base.DEFAULT_PROJECT_DIR) / "external" / "RGS_v3")
DEFAULT_SEG_CHECKPOINT = ""
DEFAULT_FILM_OUTPUT_DIR = str(Path(base.DEFAULT_PROJECT_DIR) / "checkpoints" / "film_stagewise")
DEFAULT_GEOMETRY_CSV = str(Path(base.DEFAULT_PROJECT_DIR) / "geometry_features_sanitized.csv")
DEFAULT_SEGMENTATION_MODEL_NAME = "swin_tiny_patch4_window7_224"
DEFAULT_SEGMENTATION_IMAGE_SIZE = 224
DEFAULT_SEGMENTATION_NUM_CLASSES = 3
DEFAULT_OD_CLASS = 1
DEFAULT_OC_CLASS = 0
STAGE2_SMALL_MOE_CONFIG = {
    "head_type": "moe_mtl",
    "num_experts": 2,
    "moe_dim": 128,
    "moe_hidden_dim": 256,
    "task_tower_dim": 64,
    "image_lr": 1e-5,
    "prior_lr": 1e-4,
    "head_lr": 3e-4,
    "moe_balance_loss_weight": 0.01,
    "moe_entropy_loss_weight": 0.001,
}
STRUCTURAL_PRIOR_STAGE_INDICES = (1, 2)
DEFAULT_PRIOR_HIDDEN_DIM = 32
DEFAULT_PRIOR_DROPOUT = 0.15
DEFAULT_PRIOR_DILATION_KERNEL = 15
DEFAULT_AUX_LOSS_TASKS = ("ANRS", "ANRI", "LD", "LC")
DEFAULT_SEG_MEMMAP_NAME = "od_oc_softmaps_float16.npy"
DEFAULT_SEG_MEMMAP_IDS_NAME = "od_oc_softmap_ids.npy"
DEFAULT_CROP224_SEG_CACHE_DIR = str(Path(base.DEFAULT_CACHE_DIR) / "od_oc_swin_softmaps_crop224")
CLINICAL_TASK_GROUPS = {
    "Final": "final",
    "ANRS": "rim_disc",
    "ANRI": "rim_disc",
    "LD": "lamina",
    "LC": "lamina",
    "RNFLDS": "rnfl",
    "RNFLDI": "rnfl",
    "BCLVS": "vascular",
    "BCLVI": "vascular",
    "NVT": "vascular",
    "DH": "dh",
}
TAFM_TASK_GROUPS = {
    "Final": "final",
    "ANRS": "rim_disc",
    "ANRI": "rim_disc",
    "LC": "rim_disc",
    "RNFLDS": "rnfl",
    "RNFLDI": "rnfl",
    "BCLVS": "vascular",
    "BCLVI": "vascular",
    "NVT": "vascular",
    "DH": "rare",
    "LD": "rare",
}
GCN_AUX_GROUP_SPECS = [
    ("Superior", ("DH", "BCLVS", "RNFLDS", "ANRS")),
    ("Inferior", ("DH", "BCLVI", "RNFLDI", "ANRI")),
    ("Independent", ("NVT",)),
]
GCN_AUX_PATHWAY_EDGES = [
    ("DH", "BCLVS"),
    ("BCLVS", "RNFLDS"),
    ("RNFLDS", "ANRS"),
    ("DH", "BCLVI"),
    ("BCLVI", "RNFLDI"),
    ("RNFLDI", "ANRI"),
    ("ANRS", "ANRI"),
]
FINAL_REUSE_GROUP_SCALES = {
    "Superior": 1.0,
    "Inferior": 1.0,
    "SxI interaction": 1.0,
    "Global": 0.25,
}
FINAL_REUSE_GROUP_SPECS = [
    ("Superior", ("DH", "BCLVS", "RNFLDS", "ANRS")),
    ("Inferior", ("DH", "BCLVI", "RNFLDI", "ANRI")),
    ("SxI interaction", ("DH", "BCLVS", "RNFLDS", "ANRS", "BCLVI", "RNFLDI", "ANRI")),
    ("Global", ("DH", "BCLVS", "RNFLDS", "ANRS", "BCLVI", "RNFLDI", "ANRI")),
]
DEFAULT_GEOMETRY_FEATURE_COLUMNS = [
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
    "dx_norm",
    "dy_norm",
]


def _resolve_model_name(model_name: str) -> str:
    return base.MODEL_ALIASES.get(model_name, model_name)


def _resize_map(mask: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def _normalize_probability_map(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.size == 0:
        return mask
    max_value = float(np.nanmax(mask))
    if max_value > 1.0:
        mask = mask / 255.0
    return np.clip(np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def fill_soft_ring_interior(boundary_prob: torch.Tensor) -> torch.Tensor:
    """Fill a soft closed boundary while preserving directional confidence."""
    if boundary_prob.ndim != 4 or boundary_prob.shape[1] != 1:
        raise ValueError(
            f"Expected a single-channel boundary map [B,1,H,W], got {tuple(boundary_prob.shape)}"
        )
    boundary_prob = boundary_prob.clamp(0.0, 1.0)
    left = torch.cummax(boundary_prob, dim=3).values
    right = torch.flip(
        torch.cummax(torch.flip(boundary_prob, dims=[3]), dim=3).values,
        dims=[3],
    )
    top = torch.cummax(boundary_prob, dim=2).values
    bottom = torch.flip(
        torch.cummax(torch.flip(boundary_prob, dims=[2]), dim=2).values,
        dims=[2],
    )
    enclosed_prob = torch.minimum(
        torch.minimum(left, right),
        torch.minimum(top, bottom),
    )
    return torch.maximum(boundary_prob, enclosed_prob)


def _render_softmap_rgb(softmap: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, float, float]:
    prob = _normalize_probability_map(softmap)
    prob = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
    prob_mean = float(prob.mean()) if prob.size else 0.0
    prob_max = float(prob.max()) if prob.size else 0.0
    # Keep a fixed 0-1 probability scale. Per-image max normalization can make
    # weak or spurious segmentation responses look deceptively confident.
    display = prob
    heatmap_bgr = cv2.applyColorMap(np.clip(display * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB), prob_mean, prob_max


class SwinSoftmaxODOCSegmenter:
    """Adapter for the RGS_v3 Swin-UNet segmentation checkpoint.

    It keeps the softmax probabilities as-is. No argmax/hard-mask conversion is
    applied before OD/OC maps are returned.
    """

    def __init__(
        self,
        checkpoint: str,
        rgs_v3_dir: str,
        model_name: str,
        image_size: int,
        num_classes: int,
        device: str,
        od_class: int,
        oc_class: int,
        amp_enabled: bool,
    ) -> None:
        rgs_v3_path = Path(rgs_v3_dir)
        if str(rgs_v3_path) not in sys.path:
            sys.path.insert(0, str(rgs_v3_path))

        from train_refuge_swin_unet import SwinUNet, load_checkpoint

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.od_class = int(od_class)
        self.oc_class = int(oc_class)
        self.amp_enabled = bool(amp_enabled) and self.device.type == "cuda"
        self.transform = transforms_for_swin_softmax(self.image_size)

        self.model = SwinUNet(
            model_name=model_name,
            num_classes=int(num_classes),
            image_size=self.image_size,
            pretrained=False,
        ).to(self.device)
        load_checkpoint(
            Path(checkpoint),
            model=self.model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            device=self.device,
        )
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, images: Sequence[Image.Image]) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
        if not images:
            return []
        tensors = torch.stack([self.transform(image) for image in images], dim=0).to(self.device)
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            logits = self.model(tensors)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()

        outputs: List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]] = []
        for idx, image in enumerate(images):
            probs = probabilities[idx]
            od_class = self.od_class
            oc_class = self.oc_class
            width, height = image.size
            od_map = cv2.resize(probs[od_class].astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            oc_map = cv2.resize(probs[oc_class].astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            stats = {
                "od_class": float(od_class),
                "oc_class": float(oc_class),
                "od_prob_mean": float(od_map.mean()),
                "oc_prob_mean": float(oc_map.mean()),
                "od_prob_max": float(od_map.max()),
                "oc_prob_max": float(oc_map.max()),
            }
            outputs.append((od_map, oc_map, stats))
        return outputs


def transforms_for_swin_softmax(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _seg_cache_path(cache_dir: Path, image_value: object) -> Path:
    stem = Path(str(image_value)).stem
    return cache_dir / f"{stem}.npz"


def _load_seg_cache(cache_dir: Path, image_value: object) -> Optional[Tuple[np.ndarray, np.ndarray, bool, Dict[str, float]]]:
    path = _seg_cache_path(cache_dir, image_value)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            od_map = data["od_map"].astype(np.float32)
            oc_map = data["oc_map"].astype(np.float32)
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    if od_map.ndim != 2 or oc_map.ndim != 2 or od_map.size == 0 or oc_map.size == 0:
        return None
    return od_map, oc_map, True, {}


def build_segmentation_cache(
    args: argparse.Namespace,
    dataframe: pd.DataFrame,
    target_image_ids: Optional[Sequence[object]] = None,
) -> None:
    cache_dir = Path(args.seg_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_ids = None if target_image_ids is None else {str(image_id) for image_id in target_image_ids}
    if target_ids is not None:
        dataframe = dataframe[dataframe[base.IMAGE_COL].astype(str).isin(target_ids)].copy()
        if dataframe.empty:
            raise KeyError("None of the requested segmentation cache IDs were found in the dataframe")
    image_dir, image_cache = base._load_image_backend(args)
    dataset = base.JustRAIGSMultiTaskDataset(
        dataframe=dataframe,
        image_dir=image_dir,
        image_cache=image_cache,
        transform=None,
        require_targets=False,
    )
    built = 0
    skipped = 0

    segmenter = SwinSoftmaxODOCSegmenter(
        checkpoint=args.seg_checkpoint,
        rgs_v3_dir=DEFAULT_RGS_V3_DIR,
        model_name=DEFAULT_SEGMENTATION_MODEL_NAME,
        image_size=DEFAULT_SEGMENTATION_IMAGE_SIZE,
        num_classes=DEFAULT_SEGMENTATION_NUM_CLASSES,
        device=args.device,
        od_class=DEFAULT_OD_CLASS,
        oc_class=DEFAULT_OC_CLASS,
        amp_enabled=not args.disable_amp,
    )
    pending_images: List[Image.Image] = []
    pending_image_ids: List[object] = []
    for idx in tqdm(range(len(dataset)), desc="Build Swin soft OD/OC cache"):
        image_id = dataframe.iloc[idx][base.IMAGE_COL]
        out_path = _seg_cache_path(cache_dir, image_id)
        force_target_rebuild = target_ids is not None and str(image_id) in target_ids
        if out_path.exists() and not args.overwrite_seg_cache and not force_target_rebuild:
            skipped += 1
            continue
        if image_cache is not None:
            image = dataset._array_to_rgb_image(image_cache.get(image_id, source_index=int(dataset.source_indices[idx])))
        else:
            image = dataset._load_image(dataset._resolve_image_path(image_id))
        pending_images.append(image)
        pending_image_ids.append(image_id)
        if len(pending_images) >= args.seg_infer_batch_size:
            built += _flush_swin_softmap_cache(args, cache_dir, segmenter, pending_image_ids, pending_images)
            pending_images = []
            pending_image_ids = []
    if pending_images:
        built += _flush_swin_softmap_cache(args, cache_dir, segmenter, pending_image_ids, pending_images)
    print(f"Segmentation cache -> built={built}, skipped={skipped}, dir={cache_dir}")


def ensure_segmentation_cache(args: argparse.Namespace, dataframe: pd.DataFrame) -> None:
    cache_dir = Path(args.seg_cache_dir)
    image_ids = dataframe[base.IMAGE_COL].tolist()
    missing = [image_id for image_id in image_ids if not _seg_cache_path(cache_dir, image_id).exists()]
    if args.overwrite_seg_cache or missing:
        if missing and not args.overwrite_seg_cache:
            print(
                "Segmentation softmap cache is missing entries; "
                f"building missing cache files ({len(missing)}/{len(image_ids)}) -> {cache_dir}"
            )
        elif args.overwrite_seg_cache:
            print(f"Rebuilding segmentation softmap cache with overwrite -> {cache_dir}")
        build_segmentation_cache(args, dataframe)


def _flush_swin_softmap_cache(
    args: argparse.Namespace,
    cache_dir: Path,
    segmenter: SwinSoftmaxODOCSegmenter,
    image_ids: Sequence[object],
    images: Sequence[Image.Image],
) -> int:
    outputs = segmenter.predict_batch(images)
    built = 0
    for image_id, (od_map, oc_map, soft_stats) in zip(image_ids, outputs):
        output_path = _seg_cache_path(cache_dir, image_id)
        temporary_path = output_path.with_name(f"{output_path.stem}.building{output_path.suffix}")
        if temporary_path.exists():
            temporary_path.unlink()
        np.savez_compressed(
            temporary_path,
            od_map=od_map.astype(np.float32),
            oc_map=oc_map.astype(np.float32),
            **soft_stats,
        )
        os.replace(temporary_path, output_path)
        built += 1
    return built


class InvalidSegmentationCacheError(RuntimeError):
    def __init__(self, image_ids: Sequence[str]) -> None:
        self.image_ids = list(image_ids)
        preview = ", ".join(self.image_ids[:5])
        super().__init__(
            f"Found {len(self.image_ids)} invalid segmentation NPZ file(s)"
            + (f": {preview}" if preview else "")
        )


class ODOCSoftmapMemmap:
    def __init__(self, maps_path: Path, ids_path: Path) -> None:
        self.maps_path = Path(maps_path)
        self.ids_path = Path(ids_path)
        image_ids = np.load(self.ids_path, allow_pickle=False)
        self.lookup = {str(image_id): index for index, image_id in enumerate(image_ids.tolist())}
        self._maps: Optional[np.ndarray] = None

    def _ensure_open(self) -> np.ndarray:
        if self._maps is None:
            self._maps = np.load(self.maps_path, mmap_mode="r", allow_pickle=False)
        return self._maps

    def get(self, image_id: object) -> Tuple[np.ndarray, np.ndarray]:
        key = str(image_id)
        if key not in self.lookup:
            raise KeyError(f"Softmap memmap has no entry for image_id={key}")
        maps = self._ensure_open()[self.lookup[key]]
        return maps[0].astype(np.float32), maps[1].astype(np.float32)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = None
        return state


def _seg_memmap_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    cache_dir = Path(args.seg_cache_dir)
    return cache_dir / args.seg_memmap_name, cache_dir / args.seg_memmap_ids_name


def _cleanup_stale_memmap_builds(maps_path: Path, ids_path: Path) -> None:
    patterns = (
        f"{maps_path.stem}.building*{maps_path.suffix}",
        f"{ids_path.stem}.building*{ids_path.suffix}",
    )
    for pattern in patterns:
        for temporary_path in maps_path.parent.glob(pattern):
            try:
                temporary_path.unlink()
                print(f"Removed stale segmentation memmap build -> {temporary_path}")
            except PermissionError:
                print(
                    "Stale segmentation memmap is still locked by another Python process; "
                    f"leaving it in place -> {temporary_path}"
                )


def ensure_segmentation_memmap(args: argparse.Namespace, dataframe: pd.DataFrame) -> ODOCSoftmapMemmap:
    maps_path, ids_path = _seg_memmap_paths(args)
    cache_dir = maps_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_memmap_builds(maps_path, ids_path)
    requested_ids = list(dict.fromkeys(dataframe[base.IMAGE_COL].astype(str).tolist()))

    existing_ids: List[str] = []
    cache_complete = False
    if maps_path.exists() and ids_path.exists() and not (args.rebuild_seg_memmap or args.overwrite_seg_cache):
        stored_ids = np.load(ids_path, allow_pickle=False)
        stored_maps = np.load(maps_path, mmap_mode="r", allow_pickle=False)
        existing_ids = [str(value) for value in stored_ids.tolist()]
        cache_complete = stored_maps.ndim == 4 and stored_maps.shape[0] == len(existing_ids) and stored_maps.shape[1] == 2
        del stored_maps
        if cache_complete and set(requested_ids).issubset(existing_ids):
            print(f"Using OD/OC memmap cache: {maps_path} | entries={len(existing_ids)}")
            return ODOCSoftmapMemmap(maps_path, ids_path)

    target_ids = list(dict.fromkeys([*existing_ids, *requested_ids])) if cache_complete else requested_ids
    missing_npz = [image_id for image_id in target_ids if not _seg_cache_path(cache_dir, image_id).exists()]
    if missing_npz:
        raise FileNotFoundError(
            f"Cannot build OD/OC memmap: {len(missing_npz)} source .npz files are missing. "
            f"First missing image_id={missing_npz[0]}"
        )

    process_tag = os.getpid()
    maps_tmp = maps_path.with_name(f"{maps_path.stem}.building-{process_tag}{maps_path.suffix}")
    ids_tmp = ids_path.with_name(f"{ids_path.stem}.building-{process_tag}{ids_path.suffix}")
    for temp_path in (maps_tmp, ids_tmp):
        if temp_path.exists():
            temp_path.unlink()

    print(
        f"Building consolidated OD/OC memmap from {len(target_ids)} .npz files -> {maps_path} "
        f"(float16, estimated={len(target_ids) * 2 * DEFAULT_SEGMENTATION_IMAGE_SIZE ** 2 * 2 / (1024 ** 3):.1f} GiB)"
    )
    memmap = np.lib.format.open_memmap(
        maps_tmp,
        mode="w+",
        dtype=np.float16,
        shape=(len(target_ids), 2, DEFAULT_SEGMENTATION_IMAGE_SIZE, DEFAULT_SEGMENTATION_IMAGE_SIZE),
    )
    invalid_ids: List[str] = []
    for index, image_id in enumerate(tqdm(target_ids, desc="Consolidate OD/OC softmaps")):
        cached = _load_seg_cache(cache_dir, image_id)
        if cached is None:
            invalid_ids.append(image_id)
            continue
        od_map, oc_map, _, _ = cached
        if od_map.shape != (DEFAULT_SEGMENTATION_IMAGE_SIZE, DEFAULT_SEGMENTATION_IMAGE_SIZE):
            od_map = _resize_map(od_map, DEFAULT_SEGMENTATION_IMAGE_SIZE)
        if oc_map.shape != (DEFAULT_SEGMENTATION_IMAGE_SIZE, DEFAULT_SEGMENTATION_IMAGE_SIZE):
            oc_map = _resize_map(oc_map, DEFAULT_SEGMENTATION_IMAGE_SIZE)
        memmap[index, 0] = od_map.astype(np.float16)
        memmap[index, 1] = oc_map.astype(np.float16)
    memmap.flush()
    del memmap
    if invalid_ids:
        if maps_tmp.exists():
            maps_tmp.unlink()
        raise InvalidSegmentationCacheError(invalid_ids)

    max_id_length = max((len(image_id) for image_id in target_ids), default=1)
    np.save(ids_tmp, np.asarray(target_ids, dtype=f"<U{max_id_length}"), allow_pickle=False)
    os.replace(maps_tmp, maps_path)
    os.replace(ids_tmp, ids_path)
    print(f"Saved OD/OC memmap cache -> {maps_path}")
    return ODOCSoftmapMemmap(maps_path, ids_path)


def prepare_segmentation_memmap(args: argparse.Namespace, dataframe: pd.DataFrame) -> ODOCSoftmapMemmap:
    if args.overwrite_seg_cache:
        ensure_segmentation_cache(args, dataframe)
    for _ in range(3):
        try:
            return ensure_segmentation_memmap(args, dataframe)
        except FileNotFoundError:
            ensure_segmentation_cache(args, dataframe)
        except InvalidSegmentationCacheError as error:
            print(
                "Invalid segmentation cache entries detected; "
                f"rebuilding only {len(error.image_ids)} affected file(s)."
            )
            build_segmentation_cache(args, dataframe, target_image_ids=error.image_ids)
    return ensure_segmentation_memmap(args, dataframe)


class FiLMInputTransform:
    def __init__(
        self,
        image_size: int,
        train: bool,
        softmap_input_mode: str = "od_oc",
        dilation_kernel: int = DEFAULT_PRIOR_DILATION_KERNEL,
    ) -> None:
        self.image_size = image_size
        self.train = train
        if softmap_input_mode not in {"od_oc", "od_oc_peri"}:
            raise ValueError(f"Unsupported softmap_input_mode: {softmap_input_mode}")
        if dilation_kernel < 1 or dilation_kernel % 2 == 0:
            raise ValueError("dilation_kernel must be a positive odd integer")
        self.softmap_input_mode = softmap_input_mode
        self.dilation_kernel = int(dilation_kernel)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image, od_map: np.ndarray, oc_map: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BICUBIC)
        od_map = _resize_map(od_map, self.image_size)
        oc_map = _resize_map(oc_map, self.image_size)

        if self.train:
            if torch.rand(()) < 0.5:
                image = TF.hflip(image)
                od_map = np.ascontiguousarray(np.fliplr(od_map))
                oc_map = np.ascontiguousarray(np.fliplr(oc_map))
            angle = float(torch.empty(1).uniform_(-10.0, 10.0).item())
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            matrix = cv2.getRotationMatrix2D((self.image_size / 2.0, self.image_size / 2.0), angle, 1.0)
            od_map = cv2.warpAffine(od_map, matrix, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR, borderValue=0)
            oc_map = cv2.warpAffine(oc_map, matrix, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR, borderValue=0)
            image = TF.adjust_brightness(image, float(torch.empty(1).uniform_(0.90, 1.10).item()))
            image = TF.adjust_contrast(image, float(torch.empty(1).uniform_(0.90, 1.10).item()))
            image = TF.adjust_saturation(image, float(torch.empty(1).uniform_(0.92, 1.08).item()))

        image_tensor = TF.to_tensor(image)
        image_tensor = (image_tensor - self.mean) / self.std
        od_boundary_tensor = torch.from_numpy(_normalize_probability_map(od_map)).unsqueeze(0).float()
        oc_tensor = torch.from_numpy(_normalize_probability_map(oc_map)).unsqueeze(0).float()
        od_tensor = fill_soft_ring_interior(od_boundary_tensor.unsqueeze(0)).squeeze(0)
        od_tensor = torch.maximum(od_tensor, oc_tensor)
        softmap_channels = [od_tensor, oc_tensor]
        if self.softmap_input_mode == "od_oc_peri":
            disc_tensor = od_tensor.unsqueeze(0)
            dilated_disc = F.max_pool2d(
                disc_tensor,
                kernel_size=self.dilation_kernel,
                stride=1,
                padding=self.dilation_kernel // 2,
            )
            peri_tensor = torch.relu(dilated_disc - disc_tensor).squeeze(0)
            softmap_channels.append(peri_tensor)
        seg_input = torch.cat(softmap_channels, dim=0)
        return image_tensor, seg_input


class JustRAIGSFiLMDataset(base.JustRAIGSMultiTaskDataset):
    def __init__(
        self,
        *args,
        seg_softmap_cache: Optional[ODOCSoftmapMemmap],
        transform: FiLMInputTransform,
        **kwargs,
    ) -> None:
        super().__init__(*args, transform=None, **kwargs)
        self.seg_softmap_cache = seg_softmap_cache
        self.film_transform = transform

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        item: Dict[str, torch.Tensor] = {"image_id": row[base.IMAGE_COL]}
        if self.image_cache is not None:
            image_array = self.image_cache.get(row[base.IMAGE_COL], source_index=int(self.source_indices[index]))
            image = self._array_to_rgb_image(image_array)
        else:
            image = self._load_image(self._resolve_image_path(row[base.IMAGE_COL]))

        if self.seg_softmap_cache is None:
            od_map = np.zeros((self.film_transform.image_size, self.film_transform.image_size), dtype=np.float32)
            oc_map = np.zeros_like(od_map)
        else:
            od_map, oc_map = self.seg_softmap_cache.get(row[base.IMAGE_COL])

        image_tensor, seg_input = self.film_transform(image, od_map, oc_map)
        item["image"] = image_tensor
        item["seg_input"] = seg_input
        item["seg_valid"] = torch.tensor(1.0, dtype=torch.float32)

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

        final_label = base.parse_binary_value(row[base.FINAL_COL])
        if math.isnan(final_label):
            raise ValueError(f"Missing final label at dataset index {index}")
        aux_labels: List[float] = []
        aux_masks: List[float] = []
        for label_col, mask_col in zip(base.AUX_COLUMNS, base.MASK_COLUMNS):
            raw_label = base.parse_binary_value(row[label_col])
            has_label = not math.isnan(raw_label)
            mask = base.parse_mask_value(row[mask_col], fallback=has_label) if has_label else 0.0
            aux_labels.append(0.0 if math.isnan(raw_label) else raw_label)
            aux_masks.append(mask)
        item["final"] = torch.tensor(final_label, dtype=torch.float32)
        item["aux"] = torch.tensor(aux_labels, dtype=torch.float32)
        item["aux_mask"] = torch.tensor(aux_masks, dtype=torch.float32)
        return item  # type: ignore[return-value]


class TaskGatedMoEMTLHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_aux_tasks: int,
        num_shared_experts: int = 4,
        expert_dim: int = 256,
        expert_hidden_dim: int = 512,
        tower_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_aux_tasks) + 1
        self.num_shared_experts = int(num_shared_experts)
        self.num_gate_choices = self.num_shared_experts + 1

        def make_expert() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_dim),
                nn.GELU(),
            )

        self.shared_experts = nn.ModuleList([make_expert() for _ in range(self.num_shared_experts)])
        self.task_experts = nn.ModuleList([make_expert() for _ in range(self.num_tasks)])
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Linear(in_features, self.num_gate_choices),
                )
                for _ in range(self.num_tasks)
            ]
        )
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(expert_dim),
                    nn.Linear(expert_dim, tower_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(tower_dim, 1),
                )
                for _ in range(self.num_tasks)
            ]
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_outputs = [expert(features) for expert in self.shared_experts]
        task_logits: List[torch.Tensor] = []
        task_gate_probs: List[torch.Tensor] = []

        for task_index in range(self.num_tasks):
            expert_outputs = torch.stack(
                [*shared_outputs, self.task_experts[task_index](features)],
                dim=1,
            )
            gate_probs = torch.softmax(self.gates[task_index](features), dim=-1)
            mixed = torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)
            task_logits.append(self.towers[task_index](mixed).squeeze(1))
            task_gate_probs.append(gate_probs)

        aux_logits = torch.stack(task_logits[1:], dim=1) if len(task_logits) > 1 else features.new_zeros((features.size(0), 0))
        return {
            "final_logits": task_logits[0],
            "aux_logits": aux_logits,
            "gate_probs": torch.stack(task_gate_probs, dim=1),
        }


class TaskFeatureGatedMoEMTLHead(nn.Module):
    """MoE-MTL head that consumes one feature vector per task."""

    def __init__(
        self,
        in_features: int,
        num_tasks: int,
        num_shared_experts: int = 1,
        expert_dim: int = 128,
        expert_hidden_dim: int = 256,
        tower_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.num_shared_experts = int(num_shared_experts)
        if self.num_tasks < 1:
            raise ValueError("Task-feature MoE requires at least one task.")
        if self.num_shared_experts < 1:
            raise ValueError("Task-feature MoE requires at least one shared expert.")
        self.num_gate_choices = self.num_shared_experts + 1

        def make_expert() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_dim),
                nn.GELU(),
            )

        self.shared_experts = nn.ModuleList([make_expert() for _ in range(self.num_shared_experts)])
        self.task_experts = nn.ModuleList([make_expert() for _ in range(self.num_tasks)])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, self.num_gate_choices),
            )
            for _ in range(self.num_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(expert_dim),
                nn.Linear(expert_dim, tower_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tower_dim, 1),
            )
            for _ in range(self.num_tasks)
        ])

    def forward(self, task_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if task_features.ndim != 3:
            raise ValueError(
                f"Expected task features with shape [B,T,F], got {tuple(task_features.shape)}"
            )
        if task_features.shape[1] != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} task feature columns, got {task_features.shape[1]}"
            )
        task_logits: List[torch.Tensor] = []
        task_gate_probs: List[torch.Tensor] = []
        for task_index in range(self.num_tasks):
            features = task_features[:, task_index]
            shared_outputs = [expert(features) for expert in self.shared_experts]
            expert_outputs = torch.stack(
                [*shared_outputs, self.task_experts[task_index](features)],
                dim=1,
            )
            gate_probs = torch.softmax(self.gates[task_index](features), dim=-1)
            mixed = torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)
            task_logits.append(self.towers[task_index](mixed).squeeze(1))
            task_gate_probs.append(gate_probs)

        aux_logits = (
            torch.stack(task_logits[1:], dim=1)
            if len(task_logits) > 1
            else task_features.new_zeros((task_features.size(0), 0))
        )
        return {
            "final_logits": task_logits[0],
            "aux_logits": aux_logits,
            "gate_probs": torch.stack(task_gate_probs, dim=1),
        }


class AuxReuseTaskFeatureMoEMTLHead(nn.Module):
    """Task-feature MoE where final RG reuses auxiliary task expert embeddings."""

    def __init__(
        self,
        in_features: int,
        num_tasks: int,
        num_shared_experts: int = 1,
        expert_dim: int = 128,
        expert_hidden_dim: int = 256,
        tower_dim: int = 64,
        dropout: float = 0.2,
        aux_reuse_task_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.num_aux_tasks = max(0, self.num_tasks - 1)
        self.num_shared_experts = int(num_shared_experts)
        if self.num_tasks < 2:
            raise ValueError("Aux-reuse MoE requires final plus at least one auxiliary task.")
        if self.num_shared_experts < 1:
            raise ValueError("Aux-reuse MoE requires at least one shared expert.")
        self.num_gate_choices = self.num_shared_experts + 1

        def make_expert() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_dim),
                nn.GELU(),
            )

        self.shared_experts = nn.ModuleList([make_expert() for _ in range(self.num_shared_experts)])
        self.task_experts = nn.ModuleList([make_expert() for _ in range(self.num_tasks)])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, self.num_gate_choices),
            )
            for _ in range(self.num_tasks)
        ])
        self.aux_towers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(expert_dim),
                nn.Linear(expert_dim, tower_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tower_dim, 1),
            )
            for _ in range(self.num_aux_tasks)
        ])
        self.aux_reuse_gate = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, self.num_aux_tasks),
        )
        if aux_reuse_task_indices is None:
            reuse_indices = torch.arange(self.num_aux_tasks, dtype=torch.long)
        else:
            reuse_indices = torch.as_tensor(list(aux_reuse_task_indices), dtype=torch.long)
            if reuse_indices.numel() == 0:
                raise ValueError("aux_reuse_moe requires at least one reusable auxiliary task.")
            if int(reuse_indices.min()) < 0 or int(reuse_indices.max()) >= self.num_aux_tasks:
                raise ValueError(
                    f"aux_reuse_task_indices must be in [0, {self.num_aux_tasks - 1}], got {reuse_indices.tolist()}"
                )
        self.register_buffer("aux_reuse_task_indices", reuse_indices, persistent=False)
        self.final_base_tower = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, tower_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tower_dim, 1),
        )
        self.aux_delta_tower = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, tower_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tower_dim, 1),
        )
        self.aux_delta_logit_scale = nn.Parameter(torch.tensor(-4.5951))
        with torch.no_grad():
            final_linear = self.aux_delta_tower[-1]
            if isinstance(final_linear, nn.Linear):
                final_linear.weight.zero_()
                final_linear.bias.zero_()

    def _mix_task_experts(
        self,
        features: torch.Tensor,
        task_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_outputs = [expert(features) for expert in self.shared_experts]
        expert_outputs = torch.stack(
            [*shared_outputs, self.task_experts[task_index](features)],
            dim=1,
        )
        gate_probs = torch.softmax(self.gates[task_index](features), dim=-1)
        mixed = torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)
        return mixed, gate_probs

    def forward(self, task_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if task_features.ndim != 3:
            raise ValueError(
                f"Expected task features with shape [B,T,F], got {tuple(task_features.shape)}"
            )
        if task_features.shape[1] != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} task feature columns, got {task_features.shape[1]}"
            )

        final_mixed, final_gate_probs = self._mix_task_experts(task_features[:, 0], 0)
        aux_mixed_outputs: List[torch.Tensor] = []
        aux_logits: List[torch.Tensor] = []
        task_gate_probs: List[torch.Tensor] = [final_gate_probs]

        for aux_index in range(self.num_aux_tasks):
            task_index = aux_index + 1
            mixed, gate_probs = self._mix_task_experts(task_features[:, task_index], task_index)
            aux_mixed_outputs.append(mixed)
            aux_logits.append(self.aux_towers[aux_index](mixed).squeeze(1))
            task_gate_probs.append(gate_probs)

        aux_memory = torch.stack(aux_mixed_outputs, dim=1)
        aux_reuse_logits = self.aux_reuse_gate(task_features[:, 0])
        if self.aux_reuse_task_indices.numel() < self.num_aux_tasks:
            masked_logits = aux_reuse_logits.new_full(aux_reuse_logits.shape, -1e4)
            masked_logits[:, self.aux_reuse_task_indices] = aux_reuse_logits[:, self.aux_reuse_task_indices]
            aux_reuse_logits = masked_logits
        aux_reuse_probs = torch.softmax(aux_reuse_logits, dim=-1)
        aux_context = torch.sum(aux_memory * aux_reuse_probs.unsqueeze(-1), dim=1)
        final_base_logits = self.final_base_tower(final_mixed).squeeze(1)
        aux_delta_logits = self.aux_delta_tower(aux_context).squeeze(1)
        aux_delta_scale = torch.sigmoid(self.aux_delta_logit_scale)
        final_logits = final_base_logits + aux_delta_scale * aux_delta_logits
        final_source_probs = torch.stack(
            [
                torch.ones_like(final_base_logits),
                aux_delta_scale.expand_as(final_base_logits),
            ],
            dim=1,
        )

        return {
            "final_logits": final_logits,
            "aux_logits": torch.stack(aux_logits, dim=1),
            "gate_probs": torch.stack(task_gate_probs, dim=1),
            "aux_reuse_probs": aux_reuse_probs,
            "final_source_probs": final_source_probs,
            "final_base_logits": final_base_logits,
            "aux_delta_logits": aux_delta_logits,
            "aux_delta_scale": aux_delta_scale.expand_as(final_base_logits),
        }


class AuxGATReuseTaskFeatureMoEMTLHead(AuxReuseTaskFeatureMoEMTLHead):
    """Aux-reuse MoE with graph-refined auxiliary evidence before RG reuse."""

    def __init__(
        self,
        in_features: int,
        num_tasks: int,
        num_shared_experts: int = 1,
        expert_dim: int = 128,
        expert_hidden_dim: int = 256,
        tower_dim: int = 64,
        dropout: float = 0.2,
        aux_reuse_task_indices: Optional[Sequence[int]] = None,
        graph_heads: int = 4,
        graph_temperature: float = 1.2,
        reuse_temperature: float = 0.8,
        self_attention_bias: float = 0.1,
        task_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            num_tasks=num_tasks,
            num_shared_experts=num_shared_experts,
            expert_dim=expert_dim,
            expert_hidden_dim=expert_hidden_dim,
            tower_dim=tower_dim,
            dropout=dropout,
            aux_reuse_task_indices=aux_reuse_task_indices,
        )
        self.graph_heads = int(graph_heads)
        self.graph_temperature = float(graph_temperature)
        self.reuse_temperature = float(reuse_temperature)
        self.aux_task_names = (
            [str(name) for name in task_names]
            if task_names is not None
            else [f"task_{index}" for index in range(self.num_aux_tasks)]
        )
        self.dh_task_index = self.aux_task_names.index("DH") if "DH" in self.aux_task_names else None
        if self.graph_heads < 1 or expert_dim % self.graph_heads != 0:
            raise ValueError("expert_dim must be divisible by graph_heads.")
        if self.graph_temperature <= 0.0 or self.reuse_temperature <= 0.0:
            raise ValueError("GAT temperatures must be positive.")
        if len(self.aux_task_names) != self.num_aux_tasks:
            raise ValueError(f"Expected {self.num_aux_tasks} auxiliary task names, got {len(self.aux_task_names)}.")
        self.graph_source = nn.Linear(expert_dim, expert_dim)
        self.graph_target = nn.Linear(expert_dim, expert_dim)
        self.graph_attention_vector = nn.Parameter(torch.empty(self.graph_heads, expert_dim // self.graph_heads))
        nn.init.xavier_uniform_(self.graph_attention_vector.unsqueeze(-1))
        self.graph_activation = nn.LeakyReLU(negative_slope=0.2)
        self.graph_value = nn.Linear(expert_dim, expert_dim)
        self.graph_output = nn.Sequential(
            nn.Linear(expert_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.graph_norm = nn.LayerNorm(expert_dim)
        self.graph_bias = nn.Parameter(torch.zeros(self.graph_heads, self.num_aux_tasks, self.num_aux_tasks))
        with torch.no_grad():
            diagonal = torch.arange(self.num_aux_tasks)
            self.graph_bias[:, diagonal, diagonal] = float(self_attention_bias)
        self.register_buffer(
            "graph_adjacency",
            self._build_clinical_aux_adjacency(self.aux_task_names),
            persistent=False,
        )
        self.reuse_logit_scale = nn.Parameter(torch.tensor(math.log(2.0)))
        self.reuse_task_embeddings = nn.Parameter(torch.zeros(self.num_aux_tasks, expert_dim))
        nn.init.normal_(self.reuse_task_embeddings, std=0.02)
        self.reuse_identity_projection = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, expert_dim),
            nn.GELU(),
        )
        group_names, group_pool = self._build_rg_reuse_groups(self.aux_task_names)
        self.rg_reuse_group_names = group_names
        self.register_buffer("rg_reuse_group_pool", group_pool, persistent=False)
        self.group_reuse_gate = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, len(group_names)),
        )

    @staticmethod
    def _build_clinical_aux_adjacency(task_names: Sequence[str]) -> torch.Tensor:
        index_by_name = {str(name): index for index, name in enumerate(task_names)}
        adjacency = torch.eye(len(task_names), dtype=torch.bool)
        for left_name, right_name in GCN_AUX_PATHWAY_EDGES:
            if left_name in index_by_name and right_name in index_by_name:
                left_index = index_by_name[left_name]
                right_index = index_by_name[right_name]
                adjacency[left_index, right_index] = True
                adjacency[right_index, left_index] = True
        return adjacency

    @staticmethod
    def _build_rg_reuse_groups(task_names: Sequence[str]) -> Tuple[List[str], torch.Tensor]:
        index_by_name = {str(name): index for index, name in enumerate(task_names)}
        rows: List[torch.Tensor] = []
        names: List[str] = []
        for group_name, members in FINAL_REUSE_GROUP_SPECS:
            row = torch.zeros(len(task_names), dtype=torch.float32)
            for member in members:
                if member in index_by_name:
                    row[index_by_name[member]] = 1.0
            if float(row.sum()) > 0.0:
                row = row / row.sum().clamp_min(1.0)
                names.append(group_name)
                rows.append(row)
        if not rows:
            names = ["Global"]
            rows = [torch.full((len(task_names),), 1.0 / max(1, len(task_names)), dtype=torch.float32)]
        return names, torch.stack(rows, dim=0)

    def _graph_attention(self, aux_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, feature_dim = aux_memory.shape
        head_dim = feature_dim // self.graph_heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, num_nodes, self.graph_heads, head_dim).transpose(1, 2)

        source = split_heads(self.graph_source(aux_memory))
        target = split_heads(self.graph_target(aux_memory))
        value = split_heads(self.graph_value(aux_memory))
        pair_features = self.graph_activation(source.unsqueeze(3) + target.unsqueeze(2))
        scores = torch.sum(
            pair_features * self.graph_attention_vector.view(1, self.graph_heads, 1, 1, head_dim),
            dim=-1,
        )
        scores = scores + self.graph_bias.unsqueeze(0)
        adjacency = self.graph_adjacency.to(device=scores.device).view(1, 1, num_nodes, num_nodes)
        scores = scores.masked_fill(~adjacency, -1e4)
        attention = torch.softmax(scores / self.graph_temperature, dim=-1)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(batch_size, num_nodes, feature_dim)
        return self.graph_norm(aux_memory + self.graph_output(attended)), attention

    def forward(self, task_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if task_features.ndim != 3:
            raise ValueError(
                f"Expected task features with shape [B,T,F], got {tuple(task_features.shape)}"
            )
        if task_features.shape[1] != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} task feature columns, got {task_features.shape[1]}"
            )

        final_mixed, final_gate_probs = self._mix_task_experts(task_features[:, 0], 0)
        aux_mixed_outputs: List[torch.Tensor] = []
        task_gate_probs: List[torch.Tensor] = [final_gate_probs]

        for aux_index in range(self.num_aux_tasks):
            task_index = aux_index + 1
            mixed, gate_probs = self._mix_task_experts(task_features[:, task_index], task_index)
            aux_mixed_outputs.append(mixed)
            task_gate_probs.append(gate_probs)

        aux_memory = torch.stack(aux_mixed_outputs, dim=1)
        graph_aux_memory, graph_attention = self._graph_attention(aux_memory)
        aux_logits = torch.stack(
            [
                tower(graph_aux_memory[:, aux_index]).squeeze(1)
                for aux_index, tower in enumerate(self.aux_towers)
            ],
            dim=1,
        )

        reuse_identity = self.reuse_identity_projection(self.reuse_task_embeddings).unsqueeze(0)
        reuse_nodes = graph_aux_memory + 0.25 * reuse_identity
        group_pool = self.rg_reuse_group_pool.to(device=reuse_nodes.device, dtype=reuse_nodes.dtype)
        group_nodes = torch.einsum("gt,btd->bgd", group_pool, reuse_nodes)
        group_reuse_logits = self.group_reuse_gate(task_features[:, 0])
        group_reuse_logits = group_reuse_logits - group_reuse_logits.mean(dim=-1, keepdim=True)
        group_reuse_logits = group_reuse_logits * torch.exp(self.reuse_logit_scale).clamp(0.5, 5.0)
        group_reuse_weights = torch.sigmoid(group_reuse_logits / self.reuse_temperature)
        reuse_scales = group_reuse_weights.new_tensor([
            FINAL_REUSE_GROUP_SCALES.get(group_name, 1.0)
            for group_name in self.rg_reuse_group_names
        ]).view(1, -1)
        group_reuse_weights = group_reuse_weights * reuse_scales
        if self.dh_task_index is not None:
            dh_confidence = torch.sigmoid(aux_logits[:, self.dh_task_index]).detach()
            for dh_group_name in ["Superior", "Inferior", "SxI interaction"]:
                if dh_group_name not in self.rg_reuse_group_names:
                    continue
                dh_group_index = self.rg_reuse_group_names.index(dh_group_name)
                group_reuse_weights = group_reuse_weights.clone()
                group_reuse_weights[:, dh_group_index] = (
                    group_reuse_weights[:, dh_group_index] * (0.25 + 0.75 * dh_confidence)
                )
        group_reuse_probs = group_reuse_weights / group_reuse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aux_context = torch.sum(group_nodes * group_reuse_probs.unsqueeze(-1), dim=1)

        final_base_logits = self.final_base_tower(final_mixed).squeeze(1)
        aux_delta_logits = self.aux_delta_tower(aux_context).squeeze(1)
        aux_delta_scale = torch.sigmoid(self.aux_delta_logit_scale)
        final_logits = final_base_logits + aux_delta_scale * aux_delta_logits
        final_source_probs = torch.stack(
            [
                torch.ones_like(final_base_logits),
                aux_delta_scale.expand_as(final_base_logits),
            ],
            dim=1,
        )

        return {
            "final_logits": final_logits,
            "aux_logits": aux_logits,
            "gate_probs": torch.stack(task_gate_probs, dim=1),
            "aux_reuse_probs": group_reuse_probs,
            "aux_graph_attention": graph_attention,
            "aux_graph_reuse_probs": group_reuse_probs,
            "aux_graph_reuse_group_names": self.rg_reuse_group_names,
            "final_source_probs": final_source_probs,
            "final_base_logits": final_base_logits,
            "aux_delta_logits": aux_delta_logits,
            "aux_delta_scale": aux_delta_scale.expand_as(final_base_logits),
        }


class AuxGATReuseTaskFeatureMTLHead(nn.Module):
    """Simple task-feature MTL head with clinical GATv2 group reuse for final RG."""

    def __init__(
        self,
        in_features: int,
        task_names: Sequence[str],
        graph_dim: int = 128,
        graph_heads: int = 4,
        tower_dim: int = 64,
        dropout: float = 0.2,
        graph_temperature: float = 1.2,
        reuse_temperature: float = 0.8,
        self_attention_bias: float = 0.1,
        initial_delta_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.task_names = [str(name) for name in task_names]
        if len(self.task_names) < 2:
            raise ValueError("AuxGATReuseTaskFeatureMTLHead requires final plus auxiliary tasks.")
        self.aux_task_names = self.task_names[1:]
        self.num_aux_tasks = len(self.aux_task_names)
        self.graph_dim = int(graph_dim)
        self.graph_heads = int(graph_heads)
        self.graph_temperature = float(graph_temperature)
        self.reuse_temperature = float(reuse_temperature)
        self.dh_task_index = self.aux_task_names.index("DH") if "DH" in self.aux_task_names else None
        if self.graph_dim < 1 or self.graph_heads < 1 or self.graph_dim % self.graph_heads != 0:
            raise ValueError("graph_dim must be positive and divisible by graph_heads.")
        if self.graph_temperature <= 0.0 or self.reuse_temperature <= 0.0:
            raise ValueError("GAT temperatures must be positive.")

        self.aux_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, graph_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_aux_tasks)
        ])
        self.graph_source = nn.Linear(graph_dim, graph_dim)
        self.graph_target = nn.Linear(graph_dim, graph_dim)
        self.graph_attention_vector = nn.Parameter(torch.empty(self.graph_heads, graph_dim // self.graph_heads))
        nn.init.xavier_uniform_(self.graph_attention_vector.unsqueeze(-1))
        self.graph_activation = nn.LeakyReLU(negative_slope=0.2)
        self.graph_value = nn.Linear(graph_dim, graph_dim)
        self.graph_output = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.graph_norm = nn.LayerNorm(graph_dim)
        self.graph_bias = nn.Parameter(torch.zeros(self.graph_heads, self.num_aux_tasks, self.num_aux_tasks))
        with torch.no_grad():
            diagonal = torch.arange(self.num_aux_tasks)
            self.graph_bias[:, diagonal, diagonal] = float(self_attention_bias)
        self.register_buffer(
            "graph_adjacency",
            AuxGATReuseTaskFeatureMoEMTLHead._build_clinical_aux_adjacency(self.aux_task_names),
            persistent=False,
        )

        self.aux_towers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(graph_dim),
                nn.Linear(graph_dim, tower_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tower_dim, 1),
            )
            for _ in range(self.num_aux_tasks)
        ])
        self.final_base_tower = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, tower_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tower_dim, 1),
        )
        group_names, group_pool = AuxGATReuseTaskFeatureMoEMTLHead._build_rg_reuse_groups(self.aux_task_names)
        self.rg_reuse_group_names = group_names
        self.register_buffer("rg_reuse_group_pool", group_pool, persistent=False)
        self.final_group_gate = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, len(group_names)),
        )
        self.reuse_logit_scale = nn.Parameter(torch.tensor(math.log(2.0)))
        self.reuse_task_embeddings = nn.Parameter(torch.zeros(self.num_aux_tasks, graph_dim))
        nn.init.normal_(self.reuse_task_embeddings, std=0.02)
        self.reuse_identity_projection = nn.Sequential(
            nn.LayerNorm(graph_dim),
            nn.Linear(graph_dim, graph_dim),
            nn.GELU(),
        )
        self.aux_delta_tower = nn.Sequential(
            nn.LayerNorm(graph_dim),
            nn.Linear(graph_dim, tower_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tower_dim, 1),
        )
        lambda_value = min(max(float(initial_delta_scale), 1e-4), 0.99)
        self.aux_delta_logit_scale = nn.Parameter(torch.tensor(math.log(lambda_value / (1.0 - lambda_value))))
        with torch.no_grad():
            final_linear = self.aux_delta_tower[-1]
            if isinstance(final_linear, nn.Linear):
                final_linear.weight.zero_()
                final_linear.bias.zero_()

    def _graph_attention(self, aux_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = aux_nodes.shape
        head_dim = self.graph_dim // self.graph_heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, num_nodes, self.graph_heads, head_dim).transpose(1, 2)

        source = split_heads(self.graph_source(aux_nodes))
        target = split_heads(self.graph_target(aux_nodes))
        value = split_heads(self.graph_value(aux_nodes))
        pair_features = self.graph_activation(source.unsqueeze(3) + target.unsqueeze(2))
        scores = torch.sum(
            pair_features * self.graph_attention_vector.view(1, self.graph_heads, 1, 1, head_dim),
            dim=-1,
        )
        scores = scores + self.graph_bias.unsqueeze(0)
        adjacency = self.graph_adjacency.to(device=scores.device).view(1, 1, num_nodes, num_nodes)
        scores = scores.masked_fill(~adjacency, -1e4)
        attention = torch.softmax(scores / self.graph_temperature, dim=-1)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(batch_size, num_nodes, self.graph_dim)
        return self.graph_norm(aux_nodes + self.graph_output(attended)), attention

    def forward(self, task_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if task_features.ndim != 3:
            raise ValueError(f"Expected task features with shape [B,T,F], got {tuple(task_features.shape)}")
        if task_features.shape[1] != len(self.task_names):
            raise ValueError(f"Expected {len(self.task_names)} task feature columns, got {task_features.shape[1]}")

        final_features = task_features[:, 0]
        aux_features = task_features[:, 1:]
        aux_nodes = torch.stack(
            [
                projection(aux_features[:, aux_index])
                for aux_index, projection in enumerate(self.aux_projections)
            ],
            dim=1,
        )
        graph_aux_memory, graph_attention = self._graph_attention(aux_nodes)
        aux_logits = torch.stack(
            [
                tower(graph_aux_memory[:, aux_index]).squeeze(1)
                for aux_index, tower in enumerate(self.aux_towers)
            ],
            dim=1,
        )

        reuse_identity = self.reuse_identity_projection(self.reuse_task_embeddings).unsqueeze(0)
        reuse_nodes = graph_aux_memory + 0.25 * reuse_identity
        group_pool = self.rg_reuse_group_pool.to(device=reuse_nodes.device, dtype=reuse_nodes.dtype)
        group_nodes = torch.einsum("gt,btd->bgd", group_pool, reuse_nodes)
        group_reuse_logits = self.final_group_gate(final_features)
        group_reuse_logits = group_reuse_logits - group_reuse_logits.mean(dim=-1, keepdim=True)
        group_reuse_logits = group_reuse_logits * torch.exp(self.reuse_logit_scale).clamp(0.5, 5.0)
        group_reuse_weights = torch.sigmoid(group_reuse_logits / self.reuse_temperature)
        reuse_scales = group_reuse_weights.new_tensor([
            FINAL_REUSE_GROUP_SCALES.get(group_name, 1.0)
            for group_name in self.rg_reuse_group_names
        ]).view(1, -1)
        group_reuse_weights = group_reuse_weights * reuse_scales
        if self.dh_task_index is not None:
            dh_confidence = torch.sigmoid(aux_logits[:, self.dh_task_index]).detach()
            for dh_group_name in ["Superior", "Inferior", "SxI interaction"]:
                if dh_group_name not in self.rg_reuse_group_names:
                    continue
                dh_group_index = self.rg_reuse_group_names.index(dh_group_name)
                group_reuse_weights = group_reuse_weights.clone()
                group_reuse_weights[:, dh_group_index] = (
                    group_reuse_weights[:, dh_group_index] * (0.25 + 0.75 * dh_confidence)
                )
        group_reuse_probs = group_reuse_weights / group_reuse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aux_context = torch.sum(group_nodes * group_reuse_probs.unsqueeze(-1), dim=1)

        final_base_logits = self.final_base_tower(final_features).squeeze(1)
        aux_delta_logits = self.aux_delta_tower(aux_context).squeeze(1)
        aux_delta_scale = torch.sigmoid(self.aux_delta_logit_scale)
        final_logits = final_base_logits + aux_delta_scale * aux_delta_logits
        return {
            "final_logits": final_logits,
            "aux_logits": aux_logits,
            "aux_graph_attention": graph_attention,
            "aux_graph_reuse_probs": group_reuse_probs,
            "aux_graph_reuse_group_names": self.rg_reuse_group_names,
            "final_base_logits": final_base_logits,
            "aux_delta_logits": aux_delta_logits,
            "aux_delta_scale": aux_delta_scale.expand_as(final_base_logits),
        }


class AuxGCNReuseTaskFeatureMTLHead(AuxGATReuseTaskFeatureMTLHead):
    """Simple task-feature MTL head with fixed clinical GCN group reuse for final RG."""

    def __init__(
        self,
        in_features: int,
        task_names: Sequence[str],
        graph_dim: int = 128,
        tower_dim: int = 64,
        dropout: float = 0.2,
        reuse_temperature: float = 0.8,
        initial_delta_scale: float = 0.01,
    ) -> None:
        super().__init__(
            in_features=in_features,
            task_names=task_names,
            graph_dim=graph_dim,
            graph_heads=1,
            tower_dim=tower_dim,
            dropout=dropout,
            graph_temperature=1.0,
            reuse_temperature=reuse_temperature,
            self_attention_bias=0.0,
            initial_delta_scale=initial_delta_scale,
        )
        default_adjacency = self._normalize_adjacency(
            AuxGATReuseTaskFeatureMoEMTLHead._build_clinical_aux_adjacency(self.aux_task_names).float()
        )
        self.register_buffer("gcn_adjacency", default_adjacency, persistent=True)
        self.gcn_transform = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.float()
        adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))
        diagonal = torch.arange(adjacency.shape[0], device=adjacency.device)
        adjacency[diagonal, diagonal] = 1.0
        degree = adjacency.sum(dim=1).clamp_min(1e-6)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]

    def set_graph_adjacency(self, adjacency: torch.Tensor) -> None:
        normalized = self._normalize_adjacency(adjacency.detach().float().cpu())
        if normalized.shape != self.gcn_adjacency.shape:
            raise ValueError(f"Expected adjacency shape {tuple(self.gcn_adjacency.shape)}, got {tuple(normalized.shape)}")
        self.gcn_adjacency.copy_(normalized.to(device=self.gcn_adjacency.device, dtype=self.gcn_adjacency.dtype))

    def _graph_attention(self, aux_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        adjacency = self.gcn_adjacency.to(device=aux_nodes.device, dtype=aux_nodes.dtype)
        propagated = torch.einsum("ij,bjd->bid", adjacency, aux_nodes)
        graph_nodes = self.graph_norm(aux_nodes + self.gcn_transform(propagated))
        graph_matrix = adjacency.view(1, 1, adjacency.shape[0], adjacency.shape[1]).expand(aux_nodes.size(0), 1, -1, -1)
        return graph_nodes, graph_matrix

    def forward(self, task_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = super().forward(task_features)
        outputs["aux_graph_fixed"] = True
        return outputs


class ClinicallyGroupedMoEMTLHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        task_names: Sequence[str],
        num_shared_experts: int = 1,
        expert_dim: int = 128,
        expert_hidden_dim: int = 256,
        tower_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.task_names = [str(name) for name in task_names]
        self.num_tasks = len(self.task_names)
        self.num_shared_experts = int(num_shared_experts)
        if self.num_shared_experts < 1:
            raise ValueError("Grouped MoE requires at least one shared expert.")

        missing = [name for name in self.task_names if name not in CLINICAL_TASK_GROUPS]
        if missing:
            raise ValueError(f"No clinical expert group configured for tasks: {missing}")
        self.task_group_names = [CLINICAL_TASK_GROUPS[name] for name in self.task_names]
        self.group_names = list(dict.fromkeys(self.task_group_names))
        self.num_gate_choices = self.num_shared_experts + 1

        def make_expert() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_dim),
                nn.GELU(),
            )

        self.shared_experts = nn.ModuleList([make_expert() for _ in range(self.num_shared_experts)])
        self.group_experts = nn.ModuleDict({group_name: make_expert() for group_name in self.group_names})
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, self.num_gate_choices),
            )
            for _ in self.task_names
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(expert_dim),
                nn.Linear(expert_dim, tower_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tower_dim, 1),
            )
            for _ in self.task_names
        ])

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_outputs = [expert(features) for expert in self.shared_experts]
        group_outputs = {
            group_name: expert(features)
            for group_name, expert in self.group_experts.items()
        }
        task_logits: List[torch.Tensor] = []
        task_gate_probs: List[torch.Tensor] = []

        for task_index, group_name in enumerate(self.task_group_names):
            expert_outputs = torch.stack(
                [*shared_outputs, group_outputs[group_name]],
                dim=1,
            )
            gate_probs = torch.softmax(self.gates[task_index](features), dim=-1)
            mixed = torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)
            task_logits.append(self.towers[task_index](mixed).squeeze(1))
            task_gate_probs.append(gate_probs)

        return {
            "final_logits": task_logits[0],
            "aux_logits": torch.stack(task_logits[1:], dim=1),
            "gate_probs": torch.stack(task_gate_probs, dim=1),
        }


class HybridClinicallyGroupedMoEMTLHead(nn.Module):
    """Route each task across shared, clinical-group, and task-specific experts."""

    def __init__(
        self,
        in_features: int,
        task_names: Sequence[str],
        num_shared_experts: int = 1,
        expert_dim: int = 128,
        expert_hidden_dim: int = 256,
        tower_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.task_names = [str(name) for name in task_names]
        self.num_tasks = len(self.task_names)
        self.num_shared_experts = int(num_shared_experts)
        if self.num_shared_experts < 1:
            raise ValueError("Hybrid grouped MoE requires at least one shared expert.")

        missing = [name for name in self.task_names if name not in CLINICAL_TASK_GROUPS]
        if missing:
            raise ValueError(f"No clinical expert group configured for tasks: {missing}")
        self.task_group_names = [CLINICAL_TASK_GROUPS[name] for name in self.task_names]
        self.group_names = list(dict.fromkeys(self.task_group_names))
        self.num_gate_choices = self.num_shared_experts + 2

        def make_expert() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_dim),
                nn.GELU(),
            )

        self.shared_experts = nn.ModuleList([make_expert() for _ in range(self.num_shared_experts)])
        self.group_experts = nn.ModuleDict({group_name: make_expert() for group_name in self.group_names})
        self.task_experts = nn.ModuleList([make_expert() for _ in self.task_names])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, self.num_gate_choices),
            )
            for _ in self.task_names
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(expert_dim),
                nn.Linear(expert_dim, tower_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tower_dim, 1),
            )
            for _ in self.task_names
        ])

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_outputs = [expert(features) for expert in self.shared_experts]
        group_outputs = {
            group_name: expert(features)
            for group_name, expert in self.group_experts.items()
        }
        task_logits: List[torch.Tensor] = []
        task_gate_probs: List[torch.Tensor] = []

        for task_index, group_name in enumerate(self.task_group_names):
            expert_outputs = torch.stack(
                [
                    *shared_outputs,
                    group_outputs[group_name],
                    self.task_experts[task_index](features),
                ],
                dim=1,
            )
            gate_probs = torch.softmax(self.gates[task_index](features), dim=-1)
            mixed = torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)
            task_logits.append(self.towers[task_index](mixed).squeeze(1))
            task_gate_probs.append(gate_probs)

        return {
            "final_logits": task_logits[0],
            "aux_logits": torch.stack(task_logits[1:], dim=1),
            "gate_probs": torch.stack(task_gate_probs, dim=1),
        }


def build_soft_morphological_gradient(prob: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("morphological gradient kernel must be a positive odd integer")
    dilated = F.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    eroded = -F.max_pool2d(-prob, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return torch.relu(dilated - eroded).clamp(0.0, 1.0)


def build_structural_prior(
    seg_input: torch.Tensor,
    dilation_kernel: int,
    include_morph_gradient: bool = False,
) -> torch.Tensor:
    if seg_input.ndim != 4 or seg_input.shape[1] != 2:
        raise ValueError(f"Expected OD/OC softmaps with shape [B,2,H,W], got {tuple(seg_input.shape)}")
    if dilation_kernel < 1 or dilation_kernel % 2 == 0:
        raise ValueError("prior dilation kernel must be a positive odd integer")

    od_boundary_prob = seg_input[:, 0:1].clamp(0.0, 1.0)
    oc_prob = seg_input[:, 1:2].clamp(0.0, 1.0)
    od_prob = torch.maximum(fill_soft_ring_interior(od_boundary_prob), oc_prob)
    rim_prob = torch.relu(od_prob - oc_prob)
    dilated_disc = F.max_pool2d(
        od_prob,
        kernel_size=dilation_kernel,
        stride=1,
        padding=dilation_kernel // 2,
    )
    peripapillary_prob = torch.relu(dilated_disc - od_prob)
    prior_channels = [od_prob, oc_prob, rim_prob, peripapillary_prob]
    if include_morph_gradient:
        prior_channels.extend([
            build_soft_morphological_gradient(od_prob),
            build_soft_morphological_gradient(oc_prob),
        ])
    return torch.cat(prior_channels, dim=1)


def structural_prior_channel_count(structural_prior_mode: str) -> int:
    if structural_prior_mode == "explicit3":
        return 3
    if structural_prior_mode == "derived6":
        return 6
    return 4


def structural_prior_channel_names(structural_prior_mode: str) -> List[str]:
    if structural_prior_mode == "explicit3":
        return ["OD", "OC", "Peripapillary"]
    if structural_prior_mode == "derived6":
        return ["OD", "OC", "Rim", "Peripapillary", "OD morph gradient", "OC morph gradient"]
    return ["OD", "OC", "Rim", "Peripapillary"]


def structural_prior_description(structural_prior_mode: str) -> str:
    if structural_prior_mode == "explicit3":
        return "OD+OC+peripapillary(explicit3)"
    if structural_prior_mode == "derived6":
        return "OD+OC+rim+peripapillary+ODmorphgrad+OCmorphgrad(derived6)"
    return "OD+OC+rim+peripapillary(derived4)"


class StructuralPriorFiLM(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int = 32,
        prior_channels: int = 4,
    ) -> None:
        super().__init__()
        hidden_channels = max(int(hidden_channels), 8)
        reduced_channels = max(feature_channels // 8, 16)
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.spatial_film = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.channel_film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, reduced_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(reduced_channels, feature_channels * 2, kernel_size=1),
        )
        nn.init.zeros_(self.spatial_film.weight)
        nn.init.zeros_(self.spatial_film.bias)
        nn.init.zeros_(self.channel_film[-1].weight)
        nn.init.zeros_(self.channel_film[-1].bias)

    def forward(
        self,
        feature: torch.Tensor,
        structural_prior: torch.Tensor,
        prior_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prior = F.interpolate(
            structural_prior,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prior_features = self.prior_encoder(prior)
        spatial_gamma, spatial_beta = self.spatial_film(prior_features).chunk(2, dim=1)
        channel_gamma, channel_beta = self.channel_film(prior_features).chunk(2, dim=1)
        gamma = 1.0 + torch.tanh(spatial_gamma + channel_gamma)
        beta = torch.tanh(spatial_beta + channel_beta)
        modulated = gamma * feature + beta
        if prior_valid is None:
            return modulated
        valid = prior_valid.to(dtype=feature.dtype).view(-1, 1, 1, 1)
        return feature + valid * (modulated - feature)


class UncertaintyAwareResidualModulation(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int = 32,
        prior_channels: int = 4,
        initial_lambda: float = 0.05,
    ) -> None:
        super().__init__()
        hidden_channels = max(int(hidden_channels), 8)
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.attention = nn.Conv2d(hidden_channels, feature_channels, kernel_size=1)
        lambda_value = min(max(float(initial_lambda), 1e-4), 0.99)
        self.lambda_logit = nn.Parameter(torch.tensor(math.log(lambda_value / (1.0 - lambda_value))))
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def forward(
        self,
        feature: torch.Tensor,
        structural_prior: torch.Tensor,
        prior_valid: Optional[torch.Tensor],
        seg_quality: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prior = F.interpolate(
            structural_prior,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        attention = torch.tanh(self.attention(self.prior_encoder(prior)))
        if prior_valid is None:
            valid = feature.new_ones((feature.size(0), 1, 1, 1))
        else:
            valid = prior_valid.to(dtype=feature.dtype).view(-1, 1, 1, 1)
        if seg_quality is None:
            quality = feature.new_ones((feature.size(0), 1, 1, 1))
        else:
            quality = seg_quality.to(device=feature.device, dtype=feature.dtype).view(-1, 1, 1, 1)
        scale = torch.sigmoid(self.lambda_logit)
        return feature * (1.0 + valid * quality * scale * attention)


class TaskAwareStructuralPriorFiLM(nn.Module):
    """Shared structural-prior encoder with lightweight group-specific FiLM adapters."""

    def __init__(
        self,
        feature_channels: int,
        group_names: Sequence[str],
        hidden_channels: int = 32,
        prior_channels: int = 4,
    ) -> None:
        super().__init__()
        hidden_channels = max(int(hidden_channels), 8)
        reduced_channels = max(feature_channels // 8, 16)
        self.group_names = list(dict.fromkeys(str(name) for name in group_names))
        if not self.group_names:
            raise ValueError("Task-aware FiLM requires at least one group.")
        self.prior_channels = int(prior_channels)
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )

        def make_adapter() -> nn.ModuleDict:
            spatial = nn.Conv2d(hidden_channels, 2, kernel_size=1)
            channel = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_channels, reduced_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(reduced_channels, feature_channels * 2, kernel_size=1),
            )
            nn.init.zeros_(spatial.weight)
            nn.init.zeros_(spatial.bias)
            nn.init.zeros_(channel[-1].weight)
            nn.init.zeros_(channel[-1].bias)
            return nn.ModuleDict({"spatial": spatial, "channel": channel})

        self.adapters = nn.ModuleDict({group_name: make_adapter() for group_name in self.group_names})
        self.adapter_scales = nn.ParameterDict({
            group_name: nn.Parameter(torch.tensor(0.1))
            for group_name in self.group_names
        })
        self.prior_channel_logits = nn.ParameterDict({
            group_name: nn.Parameter(self._initial_prior_channel_logits(group_name, self.prior_channels))
            for group_name in self.group_names
        })
        self.last_prior_channel_weights: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _initial_prior_channel_logits(group_name: str, prior_channels: int) -> torch.Tensor:
        if prior_channels == 3:
            defaults = {
                "final": [1.0, 1.0, 1.0],
                "rim_disc": [1.2, 1.2, 1.0],
                "rnfl": [0.8, 0.7, 1.8],
                "vascular": [0.8, 0.7, 1.4],
                "rare": [0.7, 0.7, 0.8],
            }
        elif prior_channels == 6:
            defaults = {
                "final": [1.0, 1.0, 1.0, 1.0, 0.8, 0.8],
                "rim_disc": [1.0, 1.0, 1.3, 0.6, 1.4, 1.1],
                "rnfl": [0.6, 0.4, 0.7, 2.0, 0.7, 0.4],
                "vascular": [0.7, 0.4, 0.7, 1.7, 0.7, 0.4],
                "rare": [0.8, 0.6, 0.8, 0.8, 1.2, 0.8],
            }
        else:
            defaults = {
                "final": [1.0, 1.0, 1.0, 1.0],
                "rim_disc": [1.2, 1.2, 1.5, 0.7],
                "rnfl": [0.7, 0.5, 0.8, 2.0],
                "vascular": [0.7, 0.5, 0.7, 1.6],
                "rare": [0.6, 0.5, 0.6, 0.6],
            }
        values = defaults.get(group_name, [1.0] * prior_channels)
        if len(values) != prior_channels:
            values = [1.0] * prior_channels
        values_tensor = torch.tensor(values, dtype=torch.float32).clamp_min(1e-4)
        probabilities = values_tensor / values_tensor.sum()
        return torch.log(probabilities)

    def prior_channel_weights(self) -> Dict[str, torch.Tensor]:
        return {
            group_name: torch.softmax(logits, dim=0) * float(self.prior_channels)
            for group_name, logits in self.prior_channel_logits.items()
        }

    def forward(
        self,
        feature: torch.Tensor,
        structural_prior: torch.Tensor,
        prior_valid: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        prior = F.interpolate(
            structural_prior,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        valid = (
            feature.new_ones((feature.size(0), 1, 1, 1))
            if prior_valid is None
            else prior_valid.to(dtype=feature.dtype).view(-1, 1, 1, 1)
        )

        outputs: Dict[str, torch.Tensor] = {}
        prior_weights = self.prior_channel_weights()
        self.last_prior_channel_weights = {
            group_name: weights.detach()
            for group_name, weights in prior_weights.items()
        }
        for group_name, adapter in self.adapters.items():
            weighted_prior = prior * prior_weights[group_name].view(1, -1, 1, 1)
            prior_features = self.prior_encoder(weighted_prior)
            spatial_gamma, spatial_beta = adapter["spatial"](prior_features).chunk(2, dim=1)
            channel_gamma, channel_beta = adapter["channel"](prior_features).chunk(2, dim=1)
            gamma = torch.tanh(spatial_gamma + channel_gamma)
            beta = torch.tanh(spatial_beta + channel_beta)
            scale = torch.tanh(self.adapter_scales[group_name])
            outputs[group_name] = feature + valid * scale * (gamma * feature + beta)
        return outputs


class SharedSpatialGate(nn.Module):
    """Shared stage-2 spatial gate derived from the structural prior."""

    def __init__(
        self,
        prior_channels: int,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        hidden_channels = max(int(hidden_channels), 8)
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.spatial_gate = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.gate_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)

    def forward(
        self,
        feature: torch.Tensor,
        structural_prior: torch.Tensor,
        prior_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prior = F.interpolate(
            structural_prior,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate = torch.tanh(self.spatial_gate(self.prior_encoder(prior)))
        scale = torch.tanh(self.gate_scale)
        valid = (
            feature.new_ones((feature.size(0), 1, 1, 1))
            if prior_valid is None
            else prior_valid.to(dtype=feature.dtype).view(-1, 1, 1, 1)
        )
        return feature * (1.0 + valid * scale * gate)


class TaskSpecificChannelFiLM(nn.Module):
    """Task-specific stage-3 channel FiLM from a shared prior embedding."""

    def __init__(
        self,
        feature_channels: int,
        task_names: Sequence[str],
        prior_channels: int,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        hidden_channels = max(int(hidden_channels), 8)
        reduced_channels = max(feature_channels // 8, 16)
        self.task_names = [str(name) for name in task_names]
        if not self.task_names:
            raise ValueError("Task-specific channel FiLM requires at least one task.")
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        def make_adapter() -> nn.Sequential:
            adapter = nn.Sequential(
                nn.Conv2d(hidden_channels, reduced_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(reduced_channels, feature_channels * 2, kernel_size=1),
            )
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            return adapter

        self.adapters = nn.ModuleDict({task_name: make_adapter() for task_name in self.task_names})
        self.adapter_scales = nn.ParameterDict({
            task_name: nn.Parameter(torch.tensor(0.05))
            for task_name in self.task_names
        })

    def forward(
        self,
        feature: torch.Tensor,
        structural_prior: torch.Tensor,
        prior_valid: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        prior = F.interpolate(
            structural_prior,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prior_embedding = self.prior_encoder(prior)
        valid = (
            feature.new_ones((feature.size(0), 1, 1, 1))
            if prior_valid is None
            else prior_valid.to(dtype=feature.dtype).view(-1, 1, 1, 1)
        )
        outputs: Dict[str, torch.Tensor] = {}
        for task_name, adapter in self.adapters.items():
            channel_gamma, channel_beta = adapter(prior_embedding).chunk(2, dim=1)
            gamma = torch.tanh(channel_gamma)
            beta = torch.tanh(channel_beta)
            scale = torch.tanh(self.adapter_scales[task_name])
            outputs[task_name] = feature + valid * scale * (gamma * feature + beta)
        return outputs


class StructuralPriorCrossAttention(nn.Module):
    """Inject structural-prior tokens into the final RGB feature map."""

    def __init__(
        self,
        feature_channels: int,
        prior_channels: int,
        attention_dim: int = 192,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if attention_dim < 1 or attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be positive and divisible by num_heads")
        prior_hidden = max(32, attention_dim // 2)
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(prior_channels + 2, prior_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, prior_hidden),
            nn.GELU(),
            nn.Conv2d(prior_hidden, attention_dim, kernel_size=1),
        )
        self.query_norm = nn.LayerNorm(feature_channels)
        self.query_projection = nn.Linear(feature_channels, attention_dim)
        self.query_position = nn.Linear(2, attention_dim, bias=False)
        self.key_value_norm = nn.LayerNorm(attention_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(attention_dim, feature_channels)
        self.output_dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _coordinate_grid(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(
        self,
        feature: torch.Tensor,
        structural_prior: torch.Tensor,
        prior_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, _, height, width = feature.shape
        prior = F.interpolate(
            structural_prior,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        coordinates = self._coordinate_grid(
            batch_size,
            height,
            width,
            feature.device,
            feature.dtype,
        )
        prior_tokens = self.prior_encoder(torch.cat([prior, coordinates], dim=1))
        prior_tokens = prior_tokens.flatten(2).transpose(1, 2)
        prior_tokens = self.key_value_norm(prior_tokens)

        feature_tokens = feature.flatten(2).transpose(1, 2)
        coordinate_tokens = coordinates.flatten(2).transpose(1, 2)
        query_tokens = self.query_projection(self.query_norm(feature_tokens))
        query_tokens = query_tokens + self.query_position(coordinate_tokens)
        attended, _ = self.attention(
            query_tokens,
            prior_tokens,
            prior_tokens,
            need_weights=False,
        )
        residual = self.output_projection(self.output_dropout(attended))
        residual = residual.transpose(1, 2).reshape_as(feature)
        scale = torch.tanh(self.residual_scale)
        if prior_valid is None:
            return feature + scale * residual
        valid = prior_valid.to(dtype=feature.dtype).view(-1, 1, 1, 1)
        return feature + valid * scale * residual


class StructuralPriorModulationModel(nn.Module):
    def __init__(
        self,
        image_model_name: str,
        pretrained: bool,
        aux_tasks: int,
        fusion_dim: int,
        dropout: float,
        aux_task_names: Optional[Sequence[str]] = None,
        geometry_dim: int = 0,
        geometry_hidden_dim: int = 128,
        head_type: str = "simple",
        num_experts: int = 4,
        moe_dim: int = 256,
        moe_hidden_dim: int = 512,
        task_tower_dim: int = 128,
        prior_hidden_dim: int = DEFAULT_PRIOR_HIDDEN_DIM,
        prior_dropout: float = DEFAULT_PRIOR_DROPOUT,
        prior_dilation_kernel: int = DEFAULT_PRIOR_DILATION_KERNEL,
        structural_prior_mode: str = "derived4",
        use_structural_prior: bool = True,
        structural_prior_integration: str = "film",
        tafm_grouping: str = "clinical",
        prior_attention_dim: int = 192,
        prior_attention_heads: int = 4,
        prior_attention_stage: int = 3,
        aux_delta_initial_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if head_type not in {"simple", "moe_mtl", "aux_reuse_moe", "aux_gat_reuse_moe", "aux_gat_reuse_mtl", "aux_gcn_reuse_mtl", "grouped_moe", "hybrid_grouped_moe"}:
            raise ValueError(f"Unsupported head_type: {head_type}")
        if not 0.0 <= prior_dropout < 1.0:
            raise ValueError("prior_dropout must be in [0, 1)")
        if prior_dilation_kernel < 1 or prior_dilation_kernel % 2 == 0:
            raise ValueError("prior_dilation_kernel must be a positive odd integer")
        if structural_prior_mode not in {"derived4", "derived6", "explicit3"}:
            raise ValueError(f"Unsupported structural_prior_mode: {structural_prior_mode}")
        if structural_prior_integration not in {
            "film",
            "input_concat",
            "cross_attention",
            "task_aware_film",
            "task_aware_uncertainty_film",
            "spatial_task_channel_film",
            "spatial_task_channel_uncertainty_film",
            "shared_film_aux_residual",
            "shared_film_group_residual",
        }:
            raise ValueError(
                f"Unsupported structural_prior_integration: {structural_prior_integration}"
            )
        if tafm_grouping not in {"clinical", "task"}:
            raise ValueError(f"Unsupported tafm_grouping: {tafm_grouping}")
        self.head_type = head_type
        self.prior_dropout = float(prior_dropout)
        self.prior_dilation_kernel = int(prior_dilation_kernel)
        self.structural_prior_mode = structural_prior_mode
        self.use_structural_prior = bool(use_structural_prior)
        self.structural_prior_integration = (
            structural_prior_integration if self.use_structural_prior else "none"
        )
        self.tafm_grouping = tafm_grouping
        prior_channels = structural_prior_channel_count(self.structural_prior_mode)
        image_input_channels = (
            3 + prior_channels
            if self.structural_prior_integration == "input_concat"
            else 3
        )
        self.image_encoder = timm.create_model(
            _resolve_model_name(image_model_name),
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=image_input_channels,
        )
        if not hasattr(self.image_encoder, "stem") or not hasattr(self.image_encoder, "stages"):
            raise ValueError("Structural prior modulation requires a ConvNeXt-style backbone with stem and stages.")
        image_dim = int(self.image_encoder.num_features)
        stage_channels = [int(stage.blocks[0].conv_dw.in_channels) for stage in self.image_encoder.stages]
        if prior_attention_stage < 0 or prior_attention_stage >= len(stage_channels):
            raise ValueError(
                f"prior_attention_stage must be in [0, {len(stage_channels) - 1}]"
            )
        self.prior_attention_stage = int(prior_attention_stage)
        resolved_aux_names = (
            [str(name) for name in aux_task_names]
            if aux_task_names is not None
            else [str(name) for name in base.AUX_COLUMNS]
        )
        if len(resolved_aux_names) != aux_tasks:
            raise ValueError(
                f"Expected {aux_tasks} auxiliary task names, got {len(resolved_aux_names)}"
            )
        aux_reuse_task_indices = [
            index
            for index, task_name in enumerate(resolved_aux_names)
            if task_name in DEFAULT_AUX_LOSS_TASKS
        ]
        self.task_names = ["Final", *resolved_aux_names]
        if self.tafm_grouping == "task":
            self.tafm_task_group_names = ["identity", *resolved_aux_names]
            self.tafm_group_names = list(resolved_aux_names)
        else:
            tafm_missing = [name for name in self.task_names if name not in TAFM_TASK_GROUPS]
            if tafm_missing:
                raise ValueError(f"No TaFM group configured for tasks: {tafm_missing}")
            self.tafm_task_group_names = [TAFM_TASK_GROUPS[name] for name in self.task_names]
            self.tafm_group_names = list(dict.fromkeys(self.tafm_task_group_names))
        self.prior_modulators = nn.ModuleDict()
        if self.structural_prior_integration in {"film", "shared_film_aux_residual", "shared_film_group_residual"}:
            self.prior_modulators.update(
                {
                    str(stage_index): StructuralPriorFiLM(
                        stage_channels[stage_index],
                        prior_hidden_dim,
                        prior_channels=prior_channels,
                    )
                    for stage_index in STRUCTURAL_PRIOR_STAGE_INDICES
                }
            )
        self.task_aware_film = (
            TaskAwareStructuralPriorFiLM(
                feature_channels=stage_channels[self.prior_attention_stage],
                group_names=self.tafm_group_names,
                hidden_channels=prior_hidden_dim,
                prior_channels=prior_channels,
            )
            if self.structural_prior_integration in {"task_aware_film", "task_aware_uncertainty_film"}
            else None
        )
        self.task_aware_uncertainty_modulator = (
            UncertaintyAwareResidualModulation(
                feature_channels=stage_channels[self.prior_attention_stage],
                hidden_channels=prior_hidden_dim,
                prior_channels=prior_channels,
            )
            if self.structural_prior_integration == "task_aware_uncertainty_film"
            else None
        )
        self.shared_spatial_gate = (
            SharedSpatialGate(
                prior_channels=prior_channels,
                hidden_channels=prior_hidden_dim,
            )
            if self.structural_prior_integration in {
                "spatial_task_channel_film",
                "spatial_task_channel_uncertainty_film",
            }
            else None
        )
        self.sstc_uncertainty_modulator = (
            UncertaintyAwareResidualModulation(
                feature_channels=stage_channels[2],
                hidden_channels=prior_hidden_dim,
                prior_channels=prior_channels,
            )
            if self.structural_prior_integration == "spatial_task_channel_uncertainty_film"
            else None
        )
        self.aux_residual_channel_film = (
            TaskSpecificChannelFiLM(
                feature_channels=stage_channels[-1],
                task_names=resolved_aux_names,
                prior_channels=prior_channels,
                hidden_channels=prior_hidden_dim,
            )
            if self.structural_prior_integration == "shared_film_aux_residual"
            else None
        )
        self.sgr_group_names = [
            group_name
            for group_name in self.tafm_group_names
            if group_name != "final"
        ]
        self.group_residual_channel_film = (
            TaskSpecificChannelFiLM(
                feature_channels=stage_channels[-1],
                task_names=self.sgr_group_names,
                prior_channels=prior_channels,
                hidden_channels=prior_hidden_dim,
            )
            if self.structural_prior_integration == "shared_film_group_residual"
            else None
        )
        self.task_channel_film = (
            TaskSpecificChannelFiLM(
                feature_channels=stage_channels[-1],
                task_names=self.task_names,
                prior_channels=prior_channels,
                hidden_channels=prior_hidden_dim,
            )
            if self.structural_prior_integration in {
                "spatial_task_channel_film",
                "spatial_task_channel_uncertainty_film",
            }
            else None
        )
        self.prior_cross_attention = (
            StructuralPriorCrossAttention(
                feature_channels=stage_channels[self.prior_attention_stage],
                prior_channels=prior_channels,
                attention_dim=prior_attention_dim,
                num_heads=prior_attention_heads,
                dropout=dropout,
            )
            if self.structural_prior_integration == "cross_attention"
            else None
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
        else:
            self.geometry_encoder = None
            self.geometry_hidden_dim = 0
        self.fusion = nn.Sequential(
            nn.LayerNorm(image_dim + self.geometry_hidden_dim),
            nn.Linear(image_dim + self.geometry_hidden_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.head_type in {"aux_gat_reuse_mtl", "aux_gcn_reuse_mtl"}:
            if self.structural_prior_integration not in {
                "task_aware_film",
                "task_aware_uncertainty_film",
                "spatial_task_channel_film",
                "spatial_task_channel_uncertainty_film",
                "shared_film_aux_residual",
                "shared_film_group_residual",
            }:
                raise ValueError(f"{self.head_type} requires task-specific feature streams.")
            graph_head_class = (
                AuxGCNReuseTaskFeatureMTLHead
                if self.head_type == "aux_gcn_reuse_mtl"
                else AuxGATReuseTaskFeatureMTLHead
            )
            self.task_moe_head = graph_head_class(
                in_features=fusion_dim,
                task_names=["Final", *resolved_aux_names],
                graph_dim=moe_dim,
                tower_dim=task_tower_dim,
                dropout=dropout,
                initial_delta_scale=aux_delta_initial_scale,
            )
            self.final_head = None
            self.aux_head = None
        elif self.head_type in {"moe_mtl", "aux_reuse_moe", "aux_gat_reuse_moe"}:
            if self.head_type in {"aux_reuse_moe", "aux_gat_reuse_moe"}:
                if self.structural_prior_integration not in {
                    "task_aware_film",
                    "task_aware_uncertainty_film",
                    "spatial_task_channel_film",
                    "spatial_task_channel_uncertainty_film",
                    "shared_film_aux_residual",
                    "shared_film_group_residual",
                }:
                    raise ValueError("aux_reuse_moe requires task-specific feature streams.")
                aux_reuse_head_class = (
                    AuxGATReuseTaskFeatureMoEMTLHead
                    if self.head_type == "aux_gat_reuse_moe"
                    else AuxReuseTaskFeatureMoEMTLHead
                )
                self.task_moe_head = aux_reuse_head_class(
                    in_features=fusion_dim,
                    num_tasks=1 + aux_tasks,
                    num_shared_experts=num_experts,
                    expert_dim=moe_dim,
                    expert_hidden_dim=moe_hidden_dim,
                    tower_dim=task_tower_dim,
                    dropout=dropout,
                    aux_reuse_task_indices=None if self.head_type == "aux_gat_reuse_moe" else aux_reuse_task_indices,
                    **({"task_names": resolved_aux_names} if self.head_type == "aux_gat_reuse_moe" else {}),
                )
            elif self.structural_prior_integration in {
                "task_aware_film",
                "task_aware_uncertainty_film",
                "spatial_task_channel_film",
                "spatial_task_channel_uncertainty_film",
                "shared_film_aux_residual",
                "shared_film_group_residual",
            }:
                self.task_moe_head = TaskFeatureGatedMoEMTLHead(
                    in_features=fusion_dim,
                    num_tasks=1 + aux_tasks,
                    num_shared_experts=num_experts,
                    expert_dim=moe_dim,
                    expert_hidden_dim=moe_hidden_dim,
                    tower_dim=task_tower_dim,
                    dropout=dropout,
                )
            else:
                self.task_moe_head = TaskGatedMoEMTLHead(
                    in_features=fusion_dim,
                    num_aux_tasks=aux_tasks,
                    num_shared_experts=num_experts,
                    expert_dim=moe_dim,
                    expert_hidden_dim=moe_hidden_dim,
                    tower_dim=task_tower_dim,
                    dropout=dropout,
                )
            self.final_head = None
            self.aux_head = None
        elif self.head_type in {"grouped_moe", "hybrid_grouped_moe"}:
            grouped_head_class = (
                HybridClinicallyGroupedMoEMTLHead
                if self.head_type == "hybrid_grouped_moe"
                else ClinicallyGroupedMoEMTLHead
            )
            self.task_moe_head = grouped_head_class(
                in_features=fusion_dim,
                task_names=["Final", *resolved_aux_names],
                num_shared_experts=num_experts,
                expert_dim=moe_dim,
                expert_hidden_dim=moe_hidden_dim,
                tower_dim=task_tower_dim,
                dropout=dropout,
            )
            self.final_head = None
            self.aux_head = None
        else:
            self.task_moe_head = None
            self.final_head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(dropout), nn.Linear(fusion_dim, 1))
            self.aux_head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(dropout), nn.Linear(fusion_dim, aux_tasks))

    def _prior_valid_mask(self, seg_valid: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if seg_valid is None:
            valid = torch.ones(batch_size, device=device)
        else:
            valid = seg_valid.to(device=device, dtype=torch.float32).view(batch_size)
        if self.training and self.prior_dropout > 0.0:
            keep = torch.rand(batch_size, device=device) >= self.prior_dropout
            valid = valid * keep.to(dtype=valid.dtype)
        return valid

    @staticmethod
    def _segmentation_quality(seg_input: torch.Tensor, prior_valid: torch.Tensor) -> torch.Tensor:
        if seg_input.ndim != 4 or seg_input.shape[1] < 2:
            return prior_valid
        od_peak = seg_input[:, 0].detach().float().flatten(1).amax(dim=1)
        oc_peak = seg_input[:, 1].detach().float().flatten(1).amax(dim=1)
        quality = torch.minimum(od_peak, oc_peak).clamp(0.0, 1.0)
        return quality.to(device=prior_valid.device, dtype=prior_valid.dtype) * prior_valid

    def forward_backbone_spatial(
        self,
        images: torch.Tensor,
        seg_input: torch.Tensor,
        seg_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_structural_prior:
            prior_channels = structural_prior_channel_count(self.structural_prior_mode)
            structural_prior = images.new_zeros(
                (images.shape[0], prior_channels, images.shape[-2], images.shape[-1])
            )
        elif self.structural_prior_mode == "explicit3":
            if seg_input.ndim != 4 or seg_input.shape[1] != 3:
                raise ValueError(
                    f"Expected explicit [OD, OC, peri] softmaps with shape [B,3,H,W], got {tuple(seg_input.shape)}"
                )
            structural_prior = seg_input.clamp(0.0, 1.0)
        else:
            structural_prior = build_structural_prior(
                seg_input,
                self.prior_dilation_kernel,
                include_morph_gradient=self.structural_prior_mode == "derived6",
            )
        prior_valid = (
            self._prior_valid_mask(seg_valid, images.shape[0], images.device)
            if self.use_structural_prior
            else images.new_zeros(images.shape[0])
        )
        backbone_input = (
            torch.cat([images, structural_prior], dim=1)
            if self.structural_prior_integration == "input_concat"
            else images
        )
        x = self.image_encoder.stem(backbone_input)
        for stage_index, stage in enumerate(self.image_encoder.stages):
            x = stage(x)
            modulator = self.prior_modulators[str(stage_index)] if str(stage_index) in self.prior_modulators else None
            if modulator is not None:
                x = modulator(x, structural_prior, prior_valid)
            if (
                self.prior_cross_attention is not None
                and stage_index == self.prior_attention_stage
            ):
                x = self.prior_cross_attention(x, structural_prior, prior_valid)
        return x, structural_prior, prior_valid

    def forward_task_aware_backbone(
        self,
        images: torch.Tensor,
        seg_input: torch.Tensor,
        seg_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.task_aware_film is None:
            raise RuntimeError("Task-aware FiLM module is not initialized.")
        if not self.use_structural_prior:
            prior_channels = structural_prior_channel_count(self.structural_prior_mode)
            structural_prior = images.new_zeros(
                (images.shape[0], prior_channels, images.shape[-2], images.shape[-1])
            )
        elif self.structural_prior_mode == "explicit3":
            if seg_input.ndim != 4 or seg_input.shape[1] != 3:
                raise ValueError(
                    f"Expected explicit [OD, OC, peri] softmaps with shape [B,3,H,W], got {tuple(seg_input.shape)}"
                )
            structural_prior = seg_input.clamp(0.0, 1.0)
        else:
            structural_prior = build_structural_prior(
                seg_input,
                self.prior_dilation_kernel,
                include_morph_gradient=self.structural_prior_mode == "derived6",
            )
        prior_valid = (
            self._prior_valid_mask(seg_valid, images.shape[0], images.device)
            if self.use_structural_prior
            else images.new_zeros(images.shape[0])
        )
        seg_quality = self._segmentation_quality(seg_input, prior_valid)

        x = self.image_encoder.stem(images)
        for stage_index, stage in enumerate(self.image_encoder.stages):
            x = stage(x)
            if stage_index != self.prior_attention_stage:
                continue

            grouped_features = self.task_aware_film(x, structural_prior, prior_valid)
            if "identity" in self.tafm_task_group_names:
                grouped_features["identity"] = x
            if self.task_aware_uncertainty_modulator is not None:
                grouped_features = {
                    group_name: self.task_aware_uncertainty_modulator(
                        group_feature,
                        structural_prior,
                        prior_valid,
                        seg_quality,
                    )
                    for group_name, group_feature in grouped_features.items()
                }
            remaining_stages = list(self.image_encoder.stages)[stage_index + 1:]
            spatial_by_group: Dict[str, torch.Tensor] = {}
            for group_name, group_feature in grouped_features.items():
                y = group_feature
                for remaining_stage in remaining_stages:
                    y = remaining_stage(y)
                spatial_by_group[group_name] = y
        return spatial_by_group, structural_prior, prior_valid

        raise RuntimeError(f"Task-aware FiLM stage {self.prior_attention_stage} was not reached.")

    def forward_spatial_task_channel_backbone(
        self,
        images: torch.Tensor,
        seg_input: torch.Tensor,
        seg_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.shared_spatial_gate is None or self.task_channel_film is None:
            raise RuntimeError("Shared-spatial task-channel FiLM modules are not initialized.")
        if not self.use_structural_prior:
            prior_channels = structural_prior_channel_count(self.structural_prior_mode)
            structural_prior = images.new_zeros(
                (images.shape[0], prior_channels, images.shape[-2], images.shape[-1])
            )
        elif self.structural_prior_mode == "explicit3":
            if seg_input.ndim != 4 or seg_input.shape[1] != 3:
                raise ValueError(
                    f"Expected explicit [OD, OC, peri] softmaps with shape [B,3,H,W], got {tuple(seg_input.shape)}"
                )
            structural_prior = seg_input.clamp(0.0, 1.0)
        else:
            structural_prior = build_structural_prior(
                seg_input,
                self.prior_dilation_kernel,
                include_morph_gradient=self.structural_prior_mode == "derived6",
            )
        prior_valid = (
            self._prior_valid_mask(seg_valid, images.shape[0], images.device)
            if self.use_structural_prior
            else images.new_zeros(images.shape[0])
        )
        seg_quality = self._segmentation_quality(seg_input, prior_valid)

        x = self.image_encoder.stem(images)
        for stage_index, stage in enumerate(self.image_encoder.stages):
            x = stage(x)
            if stage_index == 2:
                x = self.shared_spatial_gate(x, structural_prior, prior_valid)
                if self.sstc_uncertainty_modulator is not None:
                    x = self.sstc_uncertainty_modulator(
                        x,
                        structural_prior,
                        prior_valid,
                        seg_quality,
                    )
        task_spatial_features = self.task_channel_film(x, structural_prior, prior_valid)
        return task_spatial_features, structural_prior, prior_valid

    def forward_shared_film_aux_residual_backbone(
        self,
        images: torch.Tensor,
        seg_input: torch.Tensor,
        seg_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.aux_residual_channel_film is None:
            raise RuntimeError("Aux-residual task FiLM module is not initialized.")
        shared_spatial, structural_prior, prior_valid = self.forward_backbone_spatial(
            images,
            seg_input,
            seg_valid=seg_valid,
        )
        aux_spatial_features = self.aux_residual_channel_film(
            shared_spatial,
            structural_prior,
            prior_valid,
        )
        task_spatial_features: Dict[str, torch.Tensor] = {"Final": shared_spatial}
        task_spatial_features.update(aux_spatial_features)
        return task_spatial_features, structural_prior, prior_valid

    def forward_shared_film_group_residual_backbone(
        self,
        images: torch.Tensor,
        seg_input: torch.Tensor,
        seg_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.group_residual_channel_film is None:
            raise RuntimeError("Group-residual TaFM module is not initialized.")
        shared_spatial, structural_prior, prior_valid = self.forward_backbone_spatial(
            images,
            seg_input,
            seg_valid=seg_valid,
        )
        group_spatial_features = self.group_residual_channel_film(
            shared_spatial,
            structural_prior,
            prior_valid,
        )
        task_spatial_features: Dict[str, torch.Tensor] = {"Final": shared_spatial}
        for task_name in self.task_names[1:]:
            group_name = TAFM_TASK_GROUPS[task_name]
            task_spatial_features[task_name] = group_spatial_features[group_name]
        return task_spatial_features, structural_prior, prior_valid

    def pool_image_features(self, spatial_features: torch.Tensor) -> torch.Tensor:
        return self.image_encoder.forward_head(spatial_features, pre_logits=True)

    def fuse_features(
        self,
        image_features: torch.Tensor,
        geometry: Optional[torch.Tensor] = None,
        geometry_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_parts = [image_features]
        if self.geometry_encoder is not None:
            if geometry is None:
                geometry = image_features.new_zeros((image_features.size(0), self.geometry_dim))
            if geometry_mask is not None:
                geometry = geometry * geometry_mask
            feature_parts.append(self.geometry_encoder(geometry))
        return self.fusion(torch.cat(feature_parts, dim=1))

    def head_outputs(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.task_moe_head is not None:
            return self.task_moe_head(features)
        if self.final_head is None or self.aux_head is None:
            raise RuntimeError("Simple heads are not initialized.")
        return {
            "final_logits": self.final_head(features).squeeze(1),
            "aux_logits": self.aux_head(features),
        }

    def head_outputs_from_task_features(self, task_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.task_moe_head is not None:
            return self.task_moe_head(task_features)
        if self.final_head is None or self.aux_head is None:
            raise RuntimeError("Task-feature heads are not initialized.")
        if task_features.ndim != 3:
            raise ValueError(f"Expected task features with shape [B,T,F], got {tuple(task_features.shape)}")
        if task_features.shape[1] != len(self.task_names):
            raise ValueError(
                f"Expected {len(self.task_names)} task feature columns, got {task_features.shape[1]}"
            )
        final_logits = self.final_head(task_features[:, 0]).squeeze(1)
        aux_task_features = task_features[:, 1:]
        batch_size, num_aux_tasks, feature_dim = aux_task_features.shape
        aux_logits_all = self.aux_head(aux_task_features.reshape(batch_size * num_aux_tasks, feature_dim))
        aux_logits_all = aux_logits_all.reshape(batch_size, num_aux_tasks, num_aux_tasks)
        aux_logits = aux_logits_all.diagonal(dim1=1, dim2=2)
        return {
            "final_logits": final_logits,
            "aux_logits": aux_logits,
        }

    def forward(
        self,
        images: torch.Tensor,
        seg_input: torch.Tensor,
        seg_valid: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
        geometry_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.structural_prior_integration in {
            "spatial_task_channel_film",
            "spatial_task_channel_uncertainty_film",
            "shared_film_aux_residual",
            "shared_film_group_residual",
        }:
            if self.structural_prior_integration in {
                "spatial_task_channel_film",
                "spatial_task_channel_uncertainty_film",
            }:
                spatial_by_task, structural_prior, prior_valid = self.forward_spatial_task_channel_backbone(
                    images,
                    seg_input,
                    seg_valid=seg_valid,
                )
            elif self.structural_prior_integration == "shared_film_group_residual":
                spatial_by_task, structural_prior, prior_valid = self.forward_shared_film_group_residual_backbone(
                    images,
                    seg_input,
                    seg_valid=seg_valid,
                )
            else:
                spatial_by_task, structural_prior, prior_valid = self.forward_shared_film_aux_residual_backbone(
                    images,
                    seg_input,
                    seg_valid=seg_valid,
                )
            missing_tasks = [task_name for task_name in self.task_names if task_name not in spatial_by_task]
            if missing_tasks:
                raise RuntimeError(
                    f"Missing task-specific spatial features for tasks: {missing_tasks}"
                )
            task_image_features = torch.stack(
                [self.pool_image_features(spatial_by_task[task_name]) for task_name in self.task_names],
                dim=1,
            )
            task_features = torch.stack(
                [
                    self.fuse_features(
                        image_features=task_image_features[:, task_index],
                        geometry=geometry,
                        geometry_mask=geometry_mask,
                    )
                    for task_index in range(task_image_features.shape[1])
                ],
                dim=1,
            )
            outputs = self.head_outputs_from_task_features(task_features)
            outputs.update({
                "head_features": task_features[:, 0],
                "task_features": task_features,
                "image_features": task_image_features[:, 0],
                "task_image_features": task_image_features,
                "structural_prior": structural_prior,
                "prior_valid": prior_valid,
            })
            return outputs

        if self.structural_prior_integration in {"task_aware_film", "task_aware_uncertainty_film"}:
            spatial_by_group, structural_prior, prior_valid = self.forward_task_aware_backbone(
                images,
                seg_input,
                seg_valid=seg_valid,
            )
            pooled_by_group = {
                group_name: self.pool_image_features(spatial_features)
                for group_name, spatial_features in spatial_by_group.items()
            }
            task_image_features = torch.stack(
                [pooled_by_group[group_name] for group_name in self.tafm_task_group_names],
                dim=1,
            )
            task_features = torch.stack(
                [
                    self.fuse_features(
                        image_features=task_image_features[:, task_index],
                        geometry=geometry,
                        geometry_mask=geometry_mask,
                    )
                    for task_index in range(task_image_features.shape[1])
                ],
                dim=1,
            )
            outputs = self.head_outputs_from_task_features(task_features)
            prior_channel_weights = (
                torch.stack(
                    [
                        self.task_aware_film.last_prior_channel_weights[group_name].to(
                            device=images.device,
                            dtype=images.dtype,
                        )
                        for group_name in self.tafm_group_names
                    ],
                    dim=0,
                )
                if self.task_aware_film is not None
                else images.new_zeros((0, 0))
            )
            outputs.update({
                "head_features": task_features[:, 0],
                "task_features": task_features,
                "image_features": task_image_features[:, 0],
                "task_image_features": task_image_features,
                "structural_prior": structural_prior,
                "prior_valid": prior_valid,
                "prior_channel_weights": prior_channel_weights,
            })
            return outputs

        spatial_features, structural_prior, prior_valid = self.forward_backbone_spatial(
            images,
            seg_input,
            seg_valid=seg_valid,
        )
        image_features = self.pool_image_features(spatial_features)
        features = self.fuse_features(
            image_features=image_features,
            geometry=geometry,
            geometry_mask=geometry_mask,
        )
        outputs = self.head_outputs(features)
        outputs.update({
            "head_features": features,
            "image_features": image_features,
            "structural_prior": structural_prior,
            "prior_valid": prior_valid,
        })
        return outputs


def build_film_optimizer(model: StructuralPriorModulationModel, image_lr: float, prior_lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    parameter_groups = [
        {"params": model.image_encoder.parameters(), "lr": image_lr},
        {"params": model.fusion.parameters(), "lr": head_lr},
    ]
    prior_parameters = list(model.prior_modulators.parameters())
    if getattr(model, "task_aware_film", None) is not None:
        prior_parameters.extend(model.task_aware_film.parameters())
    if getattr(model, "task_aware_uncertainty_modulator", None) is not None:
        prior_parameters.extend(model.task_aware_uncertainty_modulator.parameters())
    if getattr(model, "shared_spatial_gate", None) is not None:
        prior_parameters.extend(model.shared_spatial_gate.parameters())
    if getattr(model, "sstc_uncertainty_modulator", None) is not None:
        prior_parameters.extend(model.sstc_uncertainty_modulator.parameters())
    if getattr(model, "task_channel_film", None) is not None:
        prior_parameters.extend(model.task_channel_film.parameters())
    if getattr(model, "aux_residual_channel_film", None) is not None:
        prior_parameters.extend(model.aux_residual_channel_film.parameters())
    if getattr(model, "group_residual_channel_film", None) is not None:
        prior_parameters.extend(model.group_residual_channel_film.parameters())
    if getattr(model, "prior_cross_attention", None) is not None:
        prior_parameters.extend(model.prior_cross_attention.parameters())
    if prior_parameters:
        parameter_groups.append({"params": prior_parameters, "lr": prior_lr})
    if getattr(model, "task_moe_head", None) is not None:
        parameter_groups.append({"params": model.task_moe_head.parameters(), "lr": head_lr})
    else:
        parameter_groups.extend([
            {"params": model.final_head.parameters(), "lr": head_lr},
            {"params": model.aux_head.parameters(), "lr": head_lr},
        ])
    if getattr(model, "geometry_encoder", None) is not None:
        parameter_groups.append({"params": model.geometry_encoder.parameters(), "lr": head_lr})
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def initialize_film_output_biases(
    model: StructuralPriorModulationModel,
    final_prior: float,
    aux_priors: torch.Tensor,
) -> None:
    if getattr(model, "task_moe_head", None) is None:
        base.initialize_output_biases(model, final_prior=final_prior, aux_priors=aux_priors)
        return

    task_moe_head = model.task_moe_head
    if task_moe_head is None:
        return
    priors = [float(final_prior)] + [float(prior) for prior in aux_priors]
    with torch.no_grad():
        if isinstance(task_moe_head, AuxReuseTaskFeatureMoEMTLHead):
            final_linear = task_moe_head.final_base_tower[-1]
            if not isinstance(final_linear, nn.Linear):
                raise TypeError("Expected aux-reuse final base tower to end with nn.Linear.")
            final_linear.bias.fill_(base._safe_logit(float(final_prior)))
            aux_delta_linear = task_moe_head.aux_delta_tower[-1]
            if not isinstance(aux_delta_linear, nn.Linear):
                raise TypeError("Expected aux-reuse aux-delta tower to end with nn.Linear.")
            aux_delta_linear.bias.zero_()
            for tower, prior in zip(task_moe_head.aux_towers, aux_priors):
                aux_linear = tower[-1]
                if not isinstance(aux_linear, nn.Linear):
                    raise TypeError("Expected each aux-reuse auxiliary tower to end with nn.Linear.")
                aux_linear.bias.fill_(base._safe_logit(float(prior)))
            return
        if isinstance(task_moe_head, AuxGATReuseTaskFeatureMTLHead):
            final_linear = task_moe_head.final_base_tower[-1]
            if not isinstance(final_linear, nn.Linear):
                raise TypeError("Expected aux-GAT MTL final base tower to end with nn.Linear.")
            final_linear.bias.fill_(base._safe_logit(float(final_prior)))
            aux_delta_linear = task_moe_head.aux_delta_tower[-1]
            if not isinstance(aux_delta_linear, nn.Linear):
                raise TypeError("Expected aux-GAT MTL aux-delta tower to end with nn.Linear.")
            aux_delta_linear.bias.zero_()
            for tower, prior in zip(task_moe_head.aux_towers, aux_priors):
                aux_linear = tower[-1]
                if not isinstance(aux_linear, nn.Linear):
                    raise TypeError("Expected each aux-GAT MTL auxiliary tower to end with nn.Linear.")
                aux_linear.bias.fill_(base._safe_logit(float(prior)))
            return
        for tower, prior in zip(task_moe_head.towers, priors):
            final_linear = tower[-1]
            if not isinstance(final_linear, nn.Linear):
                raise TypeError("Expected each MoE-MTL task tower to end with nn.Linear.")
            final_linear.bias.fill_(base._safe_logit(prior))


def warm_start_matching_checkpoint(checkpoint_path: Path, model: nn.Module, device: Optional[torch.device] = None) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    checkpoint_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    copied = 0
    partial_aux_rows = 0

    for key, tensor in checkpoint_state.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if target.shape == tensor.shape:
            target.copy_(tensor)
            copied += 1
            continue
        if key in {"aux_head.2.weight", "aux_head.2.bias"} and target.ndim == tensor.ndim:
            rows = min(target.shape[0], tensor.shape[0])
            if target.ndim == 2 and target.shape[1] == tensor.shape[1]:
                target[:rows].copy_(tensor[:rows])
                partial_aux_rows = rows
            elif target.ndim == 1:
                target[:rows].copy_(tensor[:rows])
                partial_aux_rows = rows

    model.load_state_dict(model_state)
    print(f"Warm-started FiLM model from {checkpoint_path} | copied={copied} | partial_aux_rows={partial_aux_rows}")


def summarize_auxiliary_distribution(dataframe: pd.DataFrame, split_name: str = "train") -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label_col, mask_col in zip(base.AUX_COLUMNS, base.MASK_COLUMNS):
        labels = dataframe[label_col].apply(base.parse_binary_value)
        valid_label = ~labels.isna()
        masks = pd.Series(0.0, index=dataframe.index, dtype=float)
        masks.loc[valid_label] = dataframe.loc[valid_label, mask_col].apply(
            lambda value: base.parse_mask_value(value, fallback=True)
        )
        valid = valid_label & (masks > 0)
        positives = int((labels[valid] == 1).sum())
        negatives = int((labels[valid] == 0).sum())
        valid_count = positives + negatives
        rows.append({
            "split": split_name,
            "task": label_col,
            "valid": valid_count,
            "positive": positives,
            "negative": negatives,
            "positive_rate": positives / max(valid_count, 1),
            "valid_rate": valid_count / max(len(dataframe), 1),
        })
    return pd.DataFrame(rows)


def log_auxiliary_distribution(dataframe: pd.DataFrame, split_name: str = "train") -> None:
    if not base.AUX_COLUMNS:
        return
    summary = summarize_auxiliary_distribution(dataframe, split_name=split_name)
    pieces = [
        (
            f"{row.task}: valid={int(row.valid)}({float(row.valid_rate):.1%}), "
            f"pos={int(row.positive)}, neg={int(row.negative)}, pos_rate={float(row.positive_rate):.1%}"
        )
        for row in summary.itertuples(index=False)
    ]
    print(f"Auxiliary label distribution from {split_name} split -> " + " | ".join(pieces))


def build_multitask_sample_weights(
    dataframe: pd.DataFrame,
    include_final: bool = True,
    aux_task_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    weights = np.ones(len(dataframe), dtype=np.float64)
    components = np.zeros(len(dataframe), dtype=np.float64)
    component_count = 0
    selected_aux = set(str(name) for name in aux_task_names) if aux_task_names is not None else None

    if include_final:
        final_labels = base._require_binary_labels(dataframe)
        counts = final_labels.value_counts().to_dict()
        class_weights = {label: len(final_labels) / (2.0 * max(count, 1)) for label, count in counts.items()}
        components += final_labels.map(class_weights).astype(float).to_numpy()
        component_count += 1

    for label_col, mask_col in zip(base.AUX_COLUMNS, base.MASK_COLUMNS):
        if selected_aux is not None and label_col not in selected_aux:
            continue
        labels = dataframe[label_col].apply(base.parse_binary_value)
        valid_label = ~labels.isna()
        masks = pd.Series(0.0, index=dataframe.index, dtype=float)
        masks.loc[valid_label] = dataframe.loc[valid_label, mask_col].apply(
            lambda value: base.parse_mask_value(value, fallback=True)
        )
        valid = valid_label & (masks > 0)
        positives = int((labels[valid] == 1).sum())
        negatives = int((labels[valid] == 0).sum())
        if positives + negatives <= 0:
            continue

        task_component = np.zeros(len(dataframe), dtype=np.float64)
        valid_indices = np.where(valid.to_numpy())[0]
        valid_labels = labels[valid].astype(int)
        class_counts = {0: max(negatives, 1), 1: max(positives, 1)}
        valid_weights = valid_labels.map(
            lambda label: (positives + negatives) / (2.0 * class_counts[int(label)])
        ).astype(float).to_numpy()
        task_component[valid_indices] = valid_weights
        components += task_component
        component_count += 1

    if component_count > 0:
        weights = components / float(component_count)
    weights = np.clip(weights, 1e-3, np.percentile(weights, 99.5))
    weights = weights / max(float(weights.mean()), 1e-8)
    return weights


def load_structural_prior_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    migrated_tafm_channel_weights = False
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        if getattr(model, "task_aware_film", None) is None:
            raise
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )
        allowed_missing = [
            key
            for key in missing_keys
            if key.startswith("task_aware_film.prior_channel_logits.")
        ]
        if unexpected_keys or len(allowed_missing) != len(missing_keys):
            raise exc
        migrated_tafm_channel_weights = True
        print(
            "Loaded older TaFM checkpoint without prior-channel weights; "
            "initialized channel weights from anatomy priors."
        )

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (RuntimeError, ValueError) as exc:
            if not migrated_tafm_channel_weights:
                raise
            print(f"Skipped optimizer state after TaFM channel-weight migration: {exc}")
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (RuntimeError, ValueError) as exc:
            if not migrated_tafm_channel_weights:
                raise
            print(f"Skipped scheduler state after TaFM channel-weight migration: {exc}")
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except (RuntimeError, ValueError) as exc:
            if not migrated_tafm_channel_weights:
                raise
            print(f"Skipped AMP scaler state after TaFM channel-weight migration: {exc}")

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_auroc = float(checkpoint.get("best_val_auroc", float("-inf")))
    return start_epoch, best_val_auroc


def _make_film_loader(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
    image_dir: Path,
    image_cache: Optional[base.NpyImageCache],
    seg_softmap_cache: Optional[ODOCSoftmapMemmap],
    train: bool,
    require_targets: bool = True,
) -> DataLoader:
    dataset = JustRAIGSFiLMDataset(
        dataframe=dataframe,
        image_dir=image_dir,
        image_cache=image_cache,
        transform=FiLMInputTransform(
            args.image_size,
            train=train,
            softmap_input_mode=getattr(args, "softmap_input_mode", "od_oc"),
            dilation_kernel=args.prior_dilation_kernel,
        ),
        require_targets=require_targets,
        seg_softmap_cache=seg_softmap_cache,
        geometry_feature_columns=getattr(args, "geometry_feature_columns", []),
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    sampler = None
    shuffle = train
    if train and require_targets and getattr(args, "balanced_aux_sampler", False):
        sample_weights = build_multitask_sample_weights(
            dataframe,
            include_final=not getattr(args, "disable_balanced_final_sampler", False),
            aux_task_names=getattr(args, "aux_loss_task_names", None),
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
        print(
            "Using multitask-balanced sampler "
            f"(final+aux_loss_tasks={getattr(args, 'aux_loss_task_names', 'all')}). weight min={sample_weights.min():.3f}, "
            f"mean={sample_weights.mean():.3f}, max={sample_weights.max():.3f}"
        )
    elif train and require_targets and args.balanced_final_sampler and not args.disable_balanced_final_sampler:
        labels = base._require_binary_labels(dataframe)
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


def build_film_loaders(args: argparse.Namespace):
    train_df, val_df, test_df = base.build_train_val_test_dataframes(args)
    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    use_structural_prior = bool(getattr(args, "use_structural_prior", True))
    seg_softmap_cache = prepare_segmentation_memmap(args, all_df) if use_structural_prior else None
    image_dir, image_cache = base._load_image_backend(args)
    final_pos_weight = base.compute_final_pos_weight(train_df, max_weight=args.final_pos_weight_max, power=args.pos_weight_power)
    aux_pos_weight = base.compute_aux_pos_weights(train_df, max_weight=args.aux_pos_weight_max, power=args.pos_weight_power)
    final_prior, aux_priors = base.compute_train_label_priors(train_df)
    log_auxiliary_distribution(train_df, split_name="train")
    train_loader = _make_film_loader(train_df, args, image_dir, image_cache, seg_softmap_cache, train=True, require_targets=True)
    val_loader = _make_film_loader(val_df, args, image_dir, image_cache, seg_softmap_cache, train=False, require_targets=True)
    test_loader = _make_film_loader(test_df, args, image_dir, image_cache, seg_softmap_cache, train=False, require_targets=True)
    print(
        f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} "
        f"(seed={args.seed}, val_size={args.val_size}, test_size={args.test_size})"
    )
    return train_loader, val_loader, test_loader, final_pos_weight, aux_pos_weight, final_prior, aux_priors


def build_film_eval_loader(args: argparse.Namespace) -> DataLoader:
    eval_df = base.build_eval_dataframe(args)
    seg_softmap_cache = prepare_segmentation_memmap(args, eval_df)
    image_dir, image_cache = base._load_image_backend(args)
    return _make_film_loader(
        eval_df,
        args,
        image_dir,
        image_cache,
        seg_softmap_cache,
        train=False,
        require_targets=not args.allow_unlabeled_eval,
    )


def film_stage_output_dir(base_output_dir: str, stage: str) -> Path:
    output_dir = Path(base_output_dir)
    stage_dir_name = f"stage_runs_{stage}"
    if output_dir.name != stage_dir_name:
        output_dir = output_dir / stage_dir_name
    return output_dir


def forward_film(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    images = batch["image"]
    if getattr(model, "use_channels_last", False):
        images = images.contiguous(memory_format=torch.channels_last)
    return model(
        images,
        batch["seg_input"],
        seg_valid=batch.get("seg_valid"),
        geometry=batch.get("geometry"),
        geometry_mask=batch.get("geometry_mask"),
    )


def resolve_aux_loss_task_indices(task_names: Sequence[str]) -> List[int]:
    selected = set(str(name) for name in task_names)
    indices = [index for index, name in enumerate(base.AUX_COLUMNS) if name in selected]
    missing = sorted(selected.difference(base.AUX_COLUMNS))
    if missing:
        raise ValueError(f"Aux loss task(s) not found in active AUX_COLUMNS: {missing}")
    if not indices:
        raise ValueError("At least one auxiliary loss task must be selected.")
    return indices


def compute_film_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    final_pos_weight: Optional[torch.Tensor],
    aux_pos_weight: Optional[torch.Tensor],
    aux_loss_task_indices: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    if aux_loss_task_indices is None:
        return base.compute_loss(
            outputs,
            batch,
            final_pos_weight=final_pos_weight,
            aux_pos_weight=aux_pos_weight,
        )
    aux_mask = batch["aux_mask"].clone()
    task_selector = torch.zeros(aux_mask.shape[1], device=aux_mask.device, dtype=aux_mask.dtype)
    task_selector[list(aux_loss_task_indices)] = 1.0
    loss_batch = dict(batch)
    loss_batch["aux_mask"] = aux_mask * task_selector.view(1, -1)
    return base.compute_loss(
        outputs,
        loss_batch,
        final_pos_weight=final_pos_weight,
        aux_pos_weight=aux_pos_weight,
    )


def _npmi_weight(
    yi: np.ndarray,
    yj: np.ndarray,
    smoothing: float,
    support_k: float,
) -> float:
    valid_n = int(len(yi))
    if valid_n <= 0:
        return 0.0
    n_i = float(yi.sum())
    n_j = float(yj.sum())
    n_ij = float((yi & yj).sum())
    if n_ij <= 0:
        return 0.0
    denom = float(valid_n) + 2.0 * smoothing
    p_i = (n_i + smoothing) / denom
    p_j = (n_j + smoothing) / denom
    p_ij = (n_ij + smoothing) / denom
    pmi = math.log(max(p_ij / max(p_i * p_j, 1e-12), 1e-12))
    npmi = pmi / max(-math.log(max(p_ij, 1e-12)), 1e-8)
    return max(float(npmi), 0.0) * (n_ij / (n_ij + support_k))


def compute_train_group_npmi_adjacency(
    train_df: pd.DataFrame,
    task_names: Sequence[str],
    smoothing: float = 1.0,
    support_k: float = 10.0,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    task_names = [str(name) for name in task_names]
    task_index = {name: index for index, name in enumerate(task_names)}
    num_tasks = len(task_names)

    label_by_name: Dict[str, np.ndarray] = {}
    mask_by_name: Dict[str, np.ndarray] = {}
    for task_name, label_col, mask_col in zip(base.ALL_AUX_COLUMNS, base.AUX_COLUMNS, base.MASK_COLUMNS):
        label_values = train_df[label_col].map(base.parse_binary_value).astype(float).to_numpy()
        valid_label = ~np.isnan(label_values)
        mask_values = train_df[mask_col].map(lambda value: base.parse_mask_value(value, fallback=True)).astype(float).to_numpy()
        valid = valid_label & (mask_values > 0)
        label_by_name[task_name] = np.nan_to_num(label_values, nan=0.0) > 0.5
        mask_by_name[task_name] = valid

    group_names: List[str] = []
    group_members: List[List[str]] = []
    group_label_by_name: Dict[str, np.ndarray] = {}
    group_mask_by_name: Dict[str, np.ndarray] = {}
    for group_name, raw_members in GCN_AUX_GROUP_SPECS:
        members = [member for member in raw_members if member in task_index and member in label_by_name]
        if not members:
            continue
        group_names.append(group_name)
        group_members.append(members)
        member_masks = np.stack([mask_by_name[member] for member in members], axis=0)
        member_labels = np.stack(
            [label_by_name[member] & mask_by_name[member] for member in members],
            axis=0,
        )
        group_mask_by_name[group_name] = member_masks.any(axis=0)
        group_label_by_name[group_name] = member_labels.any(axis=0)

    group_count = len(group_names)
    group_matrix = torch.eye(group_count, dtype=torch.float32)
    for i in range(group_count):
        for j in range(i + 1, group_count):
            valid = group_mask_by_name[group_names[i]] & group_mask_by_name[group_names[j]]
            weight = _npmi_weight(
                group_label_by_name[group_names[i]][valid],
                group_label_by_name[group_names[j]][valid],
                smoothing=smoothing,
                support_k=support_k,
            )
            group_matrix[i, j] = weight
            group_matrix[j, i] = weight

    group_indices_by_task: Dict[str, List[int]] = {}
    for group_index, members in enumerate(group_members):
        for member in members:
            group_indices_by_task.setdefault(member, []).append(group_index)

    adjacency = torch.eye(num_tasks, dtype=torch.float32)
    for left_name, right_name in GCN_AUX_PATHWAY_EDGES:
        if left_name not in task_index or right_name not in task_index:
            continue
        left_groups = group_indices_by_task.get(left_name, [])
        right_groups = group_indices_by_task.get(right_name, [])
        if not left_groups or not right_groups:
            continue
        weights = [
            float(group_matrix[left_group, right_group])
            for left_group in left_groups
            for right_group in right_groups
        ]
        weight = float(np.mean(weights)) if weights else 0.0
        if weight <= 0.0:
            continue
        left_index = task_index[left_name]
        right_index = task_index[right_name]
        adjacency[left_index, right_index] = weight
        adjacency[right_index, left_index] = weight
    return adjacency, group_names, group_matrix


def compute_train_npmi_adjacency(
    train_df: pd.DataFrame,
    task_names: Sequence[str],
    smoothing: float = 1.0,
    support_k: float = 10.0,
) -> torch.Tensor:
    adjacency, _, _ = compute_train_group_npmi_adjacency(
        train_df,
        task_names,
        smoothing=smoothing,
        support_k=support_k,
    )
    return adjacency


def apply_train_npmi_adjacency_if_available(model: nn.Module, train_loader: DataLoader) -> None:
    head = getattr(model, "task_moe_head", None)
    if not isinstance(head, AuxGCNReuseTaskFeatureMTLHead):
        return
    dataset = getattr(train_loader, "dataset", None)
    train_df = getattr(dataset, "df", None)
    if train_df is None:
        print("GCN adjacency: train dataframe unavailable; using clinical binary graph.")
        return
    adjacency, group_names, group_matrix = compute_train_group_npmi_adjacency(train_df, head.aux_task_names)
    head.set_graph_adjacency(adjacency)
    edge_count = int((adjacency > 0).sum().item() - adjacency.shape[0])
    group_pairs = []
    for i, left_name in enumerate(group_names):
        for j, right_name in enumerate(group_names):
            if i < j:
                group_pairs.append(f"{left_name}-{right_name}:{float(group_matrix[i, j]):.3f}")
    print(
        "GCN adjacency from train group-NPMI -> "
        f"tasks={head.aux_task_names}, positive_directed_edges={edge_count}, "
        f"groups={group_names}, group_edges=[{', '.join(group_pairs)}], "
        f"max_offdiag={float((adjacency - torch.eye(adjacency.shape[0])).max()):.4f}"
    )


def summarize_moe_gates(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    gate_probs = outputs.get("gate_probs")
    if gate_probs is None:
        return {}
    probs = gate_probs.detach().float()
    num_choices = max(int(probs.shape[-1]), 1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    normalized_entropy = entropy / max(math.log(num_choices), 1e-8)
    return {
        "moe_gate_entropy": float(normalized_entropy.mean().detach().cpu().item()),
        "moe_gate_max": float(probs.max(dim=-1).values.mean().detach().cpu().item()),
        "moe_task_expert_usage": float(probs[..., -1].mean().detach().cpu().item()),
    }


def summarize_uncertainty_modulation(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    modulators = [
        getattr(model, "task_aware_uncertainty_modulator", None),
        getattr(model, "sstc_uncertainty_modulator", None),
    ]
    active_modulators = [module for module in modulators if module is not None]
    if not active_modulators:
        return {}

    seg_input = batch.get("seg_input")
    prior_valid = outputs.get("prior_valid", batch.get("seg_valid"))
    if seg_input is None or prior_valid is None:
        return {}
    if seg_input.ndim != 4 or seg_input.shape[1] < 2:
        return {}

    with torch.no_grad():
        od_peak = seg_input[:, 0].detach().float().flatten(1).amax(dim=1)
        oc_peak = seg_input[:, 1].detach().float().flatten(1).amax(dim=1)
        valid = prior_valid.detach().float().view(-1).to(device=seg_input.device)
        quality = torch.minimum(od_peak, oc_peak).clamp(0.0, 1.0) * valid
        lambda_values = [
            torch.sigmoid(module.lambda_logit.detach()).float()
            for module in active_modulators
            if hasattr(module, "lambda_logit")
        ]
        lambda_mean = (
            torch.stack(lambda_values).mean()
            if lambda_values
            else quality.new_tensor(float("nan"))
        )
        return {
            "seg_q_mean": float(quality.mean().detach().cpu().item()),
            "seg_q_std": float(quality.std(unbiased=False).detach().cpu().item()),
            "seg_q_min": float(quality.min().detach().cpu().item()),
            "seg_q_max": float(quality.max().detach().cpu().item()),
            "uncertainty_lambda": float(lambda_mean.detach().cpu().item()),
        }


def compute_moe_gate_regularization(
    outputs: Dict[str, torch.Tensor],
    balance_weight: float,
    entropy_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total_loss = outputs["final_logits"].new_tensor(0.0)
    stats: Dict[str, float] = {}

    gate_probs = outputs.get("gate_probs")
    if gate_probs is not None:
        probs = gate_probs.float()
        num_choices = max(int(probs.shape[-1]), 1)
        task_mean_usage = probs.mean(dim=0)
        uniform = torch.full_like(task_mean_usage, 1.0 / float(num_choices))
        balance_loss = torch.mean((task_mean_usage - uniform) ** 2)
        normalized_entropy = (
            -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
            / max(math.log(num_choices), 1e-8)
        )
        entropy_penalty = 1.0 - normalized_entropy
        gate_loss = (balance_weight * balance_loss) + (entropy_weight * entropy_penalty)
        total_loss = total_loss + gate_loss
        stats.update(summarize_moe_gates(outputs))
        stats["moe_balance_loss"] = float(balance_loss.detach().cpu().item())
        stats["moe_entropy_penalty"] = float(entropy_penalty.detach().cpu().item())

    graph_attention = outputs.get("aux_graph_attention")
    if graph_attention is not None and not bool(outputs.get("aux_graph_fixed", False)):
        edge_probs = graph_attention.float()
        num_nodes = max(int(edge_probs.shape[-1]), 1)
        target_usage = edge_probs.mean(dim=(0, 1, 2))
        target_uniform = torch.full_like(target_usage, 1.0 / float(num_nodes))
        graph_balance_loss = torch.mean((target_usage - target_uniform) ** 2)
        graph_entropy = (
            -(edge_probs * torch.log(edge_probs.clamp_min(1e-8))).sum(dim=-1).mean()
            / max(math.log(num_nodes), 1e-8)
        )
        graph_entropy_penalty = 1.0 - graph_entropy
        graph_diagonal_penalty = edge_probs.diagonal(dim1=-2, dim2=-1).mean()
        graph_loss = (
            (balance_weight * graph_balance_loss)
            + (entropy_weight * graph_entropy_penalty)
            + (0.5 * balance_weight * graph_diagonal_penalty)
        )
        total_loss = total_loss + graph_loss
        stats.update({
            "aux_graph_edge_entropy": float(graph_entropy.detach().cpu().item()),
            "aux_graph_edge_max": float(edge_probs.max(dim=-1).values.mean().detach().cpu().item()),
            "aux_graph_target_balance_loss": float(graph_balance_loss.detach().cpu().item()),
            "aux_graph_entropy_penalty": float(graph_entropy_penalty.detach().cpu().item()),
            "aux_graph_diagonal_penalty": float(graph_diagonal_penalty.detach().cpu().item()),
        })

    reuse_probs = outputs.get("aux_graph_reuse_probs")
    if reuse_probs is not None:
        reuse = reuse_probs.float()
        num_reuse_nodes = max(int(reuse.shape[-1]), 1)
        reuse_entropy = (
            -(reuse * torch.log(reuse.clamp_min(1e-8))).sum(dim=-1).mean()
            / max(math.log(num_reuse_nodes), 1e-8)
        )
        reuse_entropy_penalty = 1.0 - reuse_entropy
        reuse_max_penalty = torch.clamp(reuse.max(dim=-1).values - 0.50, min=0.0).mean()
        reuse_loss = (entropy_weight * reuse_entropy_penalty) + (balance_weight * reuse_max_penalty)
        total_loss = total_loss + reuse_loss
        stats.update({
            "aux_graph_reuse_entropy": float(reuse_entropy.detach().cpu().item()),
            "aux_graph_reuse_max": float(reuse.max(dim=-1).values.mean().detach().cpu().item()),
            "aux_graph_reuse_entropy_penalty": float(reuse_entropy_penalty.detach().cpu().item()),
            "aux_graph_reuse_max_penalty": float(reuse_max_penalty.detach().cpu().item()),
        })

    if not stats:
        return total_loss, {}
    stats["moe_gate_loss"] = float(total_loss.detach().cpu().item())
    return total_loss, stats


def train_one_epoch_film(
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
    aux_loss_task_indices: Optional[Sequence[int]] = None,
    moe_balance_loss_weight: float = 0.0,
    moe_entropy_loss_weight: float = 0.0,
) -> Dict[str, float]:
    model.train()
    loss_meter = base.AverageMeter()
    final_loss_meter = base.AverageMeter()
    aux_loss_meter = base.AverageMeter()
    seg_valid_meter = base.AverageMeter()
    gate_loss_meter = base.AverageMeter()
    gate_entropy_meter = base.AverageMeter()
    gate_max_meter = base.AverageMeter()
    task_expert_usage_meter = base.AverageMeter()
    seg_q_mean_meter = base.AverageMeter()
    seg_q_std_meter = base.AverageMeter()
    seg_q_min_meter = base.AverageMeter()
    seg_q_max_meter = base.AverageMeter()
    uncertainty_lambda_meter = base.AverageMeter()

    progress = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for batch in progress:
        batch = base.move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = forward_film(model, batch)
            losses = compute_film_loss(
                outputs,
                batch,
                final_pos_weight=final_pos_weight,
                aux_pos_weight=aux_pos_weight,
                aux_loss_task_indices=aux_loss_task_indices,
            )
            gate_loss, gate_stats = compute_moe_gate_regularization(
                outputs,
                balance_weight=moe_balance_loss_weight,
                entropy_weight=moe_entropy_loss_weight,
            )
            uncertainty_stats = summarize_uncertainty_modulation(model, batch, outputs)
            losses["loss"] = losses["loss"] + gate_loss
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
        seg_valid_meter.update(float(batch["seg_valid"].detach().mean()), batch_size)
        if gate_stats:
            gate_loss_meter.update(gate_stats["moe_gate_loss"], batch_size)
            if "moe_gate_entropy" in gate_stats:
                gate_entropy_meter.update(gate_stats["moe_gate_entropy"], batch_size)
                gate_max_meter.update(gate_stats["moe_gate_max"], batch_size)
                task_expert_usage_meter.update(gate_stats["moe_task_expert_usage"], batch_size)
        if uncertainty_stats:
            seg_q_mean_meter.update(uncertainty_stats["seg_q_mean"], batch_size)
            seg_q_std_meter.update(uncertainty_stats["seg_q_std"], batch_size)
            seg_q_min_meter.update(uncertainty_stats["seg_q_min"], batch_size)
            seg_q_max_meter.update(uncertainty_stats["seg_q_max"], batch_size)
            uncertainty_lambda_meter.update(uncertainty_stats["uncertainty_lambda"], batch_size)
        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            final=f"{final_loss_meter.avg:.4f}",
            aux=f"{aux_loss_meter.avg:.4f}",
            maps=f"{seg_valid_meter.avg:.3f}",
            q=f"{seg_q_mean_meter.avg:.3f}" if seg_q_mean_meter.count > 0 else "nan",
        )

    metrics = {
        "train_loss": loss_meter.avg,
        "train_loss_final": final_loss_meter.avg,
        "train_loss_aux": aux_loss_meter.avg,
        "train_seg_valid_rate": seg_valid_meter.avg,
    }
    if seg_q_mean_meter.count > 0:
        metrics.update({
            "train_seg_q_mean": seg_q_mean_meter.avg,
            "train_seg_q_std": seg_q_std_meter.avg,
            "train_seg_q_min": seg_q_min_meter.avg,
            "train_seg_q_max": seg_q_max_meter.avg,
            "train_uncertainty_lambda": uncertainty_lambda_meter.avg,
        })
    if gate_entropy_meter.count > 0:
        metrics.update({
            "train_moe_gate_loss": gate_loss_meter.avg,
            "train_moe_gate_entropy": gate_entropy_meter.avg,
            "train_moe_gate_max": gate_max_meter.avg,
            "train_moe_task_expert_usage": task_expert_usage_meter.avg,
        })
    elif gate_loss_meter.count > 0:
        metrics["train_moe_gate_loss"] = gate_loss_meter.avg
    return metrics


def find_best_threshold(y_true, y_prob, mode="sensitivity", target_sensitivity=0.95, fixed_threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if mode == "fixed":
        return float(fixed_threshold)
    if len(np.unique(y_true)) < 2:
        return 0.5
    if mode == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return float(thresholds[int(np.argmax(tpr - fpr))])
    if mode == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        return float(thresholds[int(np.argmax(f1))])
    if mode == "sensitivity":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        valid = np.where(tpr >= target_sensitivity)[0]
        if len(valid) == 0:
            return 0.5
        return float(thresholds[valid[np.argmin(fpr[valid])]])
    return 0.5


def threshold_summary(y_true, y_prob, threshold: float) -> str:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = y_prob >= threshold
    positives = y_true == 1
    negatives = y_true == 0
    sensitivity = float(np.sum(y_pred & positives) / max(1, np.sum(positives)))
    specificity = float(np.sum((~y_pred) & negatives) / max(1, np.sum(negatives)))
    return f"threshold={threshold:.4f} | sensitivity={sensitivity:.4f} | specificity={specificity:.4f}"


def save_aux_gat_diagnostics(
    edge_attention: np.ndarray,
    node_reuse: np.ndarray,
    task_names: Sequence[str],
    output_dir: Path,
    split_name: str,
    epoch: int,
    graph_fixed: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reuse_names = list(task_names)
    if len(node_reuse) != len(reuse_names):
        reuse_names = (
            ["Superior", "Inferior", "SxI interaction", "Global"]
            if len(node_reuse) == 4
            else [f"Reuse {index + 1}" for index in range(len(node_reuse))]
        )
    tag = f"{split_name}_epoch_{epoch:03d}" if epoch > 0 else split_name
    edge_df = pd.DataFrame(edge_attention, index=task_names, columns=task_names)
    node_df = pd.DataFrame({"task": reuse_names, "final_reuse_prob": node_reuse})
    edge_csv = output_dir / f"{tag}_edge_attention.csv"
    node_csv = output_dir / f"{tag}_node_reuse.csv"
    edge_df.to_csv(edge_csv)
    node_df.to_csv(node_csv, index=False)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Saved aux graph diagnostics CSV -> {edge_csv}, {node_csv} (plot skipped: {exc})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), gridspec_kw={"width_ratios": [1.25, 0.75]})
    image = axes[0].imshow(edge_attention, cmap="magma", vmin=0.0, vmax=max(1e-6, float(edge_attention.max())))
    if graph_fixed:
        axes[0].set_title("Aux clinical GCN adjacency")
        axes[0].set_xlabel("Neighbor task")
        axes[0].set_ylabel("Updated task")
    else:
        axes[0].set_title("Aux GAT edge attention")
        axes[0].set_xlabel("Target/key task")
        axes[0].set_ylabel("Source/query task")
    axes[0].set_xticks(np.arange(len(task_names)))
    axes[0].set_yticks(np.arange(len(task_names)))
    axes[0].set_xticklabels(task_names, rotation=45, ha="right")
    axes[0].set_yticklabels(task_names)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    order = np.argsort(node_reuse)
    ordered_tasks = [reuse_names[idx] for idx in order]
    ordered_values = node_reuse[order]
    axes[1].barh(ordered_tasks, ordered_values, color="#4C78A8")
    axes[1].set_xlim(0.0, max(1.0, float(node_reuse.max()) * 1.05))
    axes[1].set_title("Final RG clinical evidence reuse")
    axes[1].set_xlabel("Mean normalized weight")
    for y_idx, value in enumerate(ordered_values):
        axes[1].text(float(value) + 0.01, y_idx, f"{value:.3f}", va="center", fontsize=8)

    fig.suptitle(f"{split_name} aux graph diagnostics" + (f" epoch {epoch:03d}" if epoch > 0 else ""))
    fig.tight_layout()
    png_path = output_dir / f"{tag}_aux_gat_graph.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Saved aux graph diagnostics -> {png_path}")


@torch.no_grad()
def validate_film(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    final_pos_weight: Optional[torch.Tensor] = None,
    aux_pos_weight: Optional[torch.Tensor] = None,
    final_threshold=0.5,
    aux_loss_task_indices: Optional[Sequence[int]] = None,
    graph_diagnostics_dir: Optional[Path] = None,
    graph_split_name: str = "val",
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    model.eval()
    has_targets = bool(getattr(loader.dataset, "require_targets", True))
    loss_meter = base.AverageMeter()
    final_loss_meter = base.AverageMeter()
    aux_loss_meter = base.AverageMeter()
    seg_valid_meter = base.AverageMeter()
    gate_entropy_meter = base.AverageMeter()
    gate_max_meter = base.AverageMeter()
    task_expert_usage_meter = base.AverageMeter()
    seg_q_mean_meter = base.AverageMeter()
    seg_q_std_meter = base.AverageMeter()
    seg_q_min_meter = base.AverageMeter()
    seg_q_max_meter = base.AverageMeter()
    uncertainty_lambda_meter = base.AverageMeter()
    graph_edge_entropy_meter = base.AverageMeter()
    graph_edge_max_meter = base.AverageMeter()
    graph_reuse_entropy_meter = base.AverageMeter()
    graph_reuse_max_meter = base.AverageMeter()
    graph_edge_sum: Optional[torch.Tensor] = None
    graph_reuse_sum: Optional[torch.Tensor] = None
    graph_fixed = False
    graph_sample_count = 0
    final_targets: List[float] = []
    final_probs: List[float] = []
    aux_targets: List[List[float]] = [[] for _ in base.AUX_COLUMNS]
    aux_probs: List[List[float]] = [[] for _ in base.AUX_COLUMNS]
    pred_rows: List[Dict[str, object]] = []

    progress = tqdm(loader, desc=f"Valid {epoch}", leave=False)
    for batch in progress:
        batch = base.move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = forward_film(model, batch)
            if has_targets:
                losses = compute_film_loss(
                    outputs,
                    batch,
                    final_pos_weight=final_pos_weight,
                    aux_pos_weight=aux_pos_weight,
                    aux_loss_task_indices=aux_loss_task_indices,
                )

        batch_size = batch["image"].size(0)
        batch_probs_final = torch.sigmoid(outputs["final_logits"]).detach().cpu().numpy()
        batch_probs_aux = torch.sigmoid(outputs["aux_logits"]).detach().cpu().numpy()
        seg_valid_meter.update(float(batch["seg_valid"].detach().mean()), batch_size)
        uncertainty_stats = summarize_uncertainty_modulation(model, batch, outputs)
        if uncertainty_stats:
            seg_q_mean_meter.update(uncertainty_stats["seg_q_mean"], batch_size)
            seg_q_std_meter.update(uncertainty_stats["seg_q_std"], batch_size)
            seg_q_min_meter.update(uncertainty_stats["seg_q_min"], batch_size)
            seg_q_max_meter.update(uncertainty_stats["seg_q_max"], batch_size)
            uncertainty_lambda_meter.update(uncertainty_stats["uncertainty_lambda"], batch_size)
        gate_stats = summarize_moe_gates(outputs)
        if gate_stats:
            gate_entropy_meter.update(gate_stats["moe_gate_entropy"], batch_size)
            gate_max_meter.update(gate_stats["moe_gate_max"], batch_size)
            task_expert_usage_meter.update(gate_stats["moe_task_expert_usage"], batch_size)
        graph_attention = outputs.get("aux_graph_attention")
        graph_reuse_probs = outputs.get("aux_graph_reuse_probs")
        if graph_attention is not None and graph_reuse_probs is not None:
            graph_fixed = graph_fixed or bool(outputs.get("aux_graph_fixed", False))
            edge_probs = graph_attention.detach().float().cpu()
            reuse_probs_tensor = graph_reuse_probs.detach().float().cpu()
            task_count = edge_probs.size(-1)
            edge_entropy = -(edge_probs.clamp_min(1e-8) * edge_probs.clamp_min(1e-8).log()).sum(dim=-1)
            edge_entropy = edge_entropy / max(1e-8, math.log(float(task_count)))
            reuse_entropy = -(reuse_probs_tensor.clamp_min(1e-8) * reuse_probs_tensor.clamp_min(1e-8).log()).sum(dim=-1)
            reuse_entropy = reuse_entropy / max(1e-8, math.log(float(task_count)))
            graph_edge_entropy_meter.update(float(edge_entropy.mean()), batch_size)
            graph_edge_max_meter.update(float(edge_probs.max(dim=-1).values.mean()), batch_size)
            graph_reuse_entropy_meter.update(float(reuse_entropy.mean()), batch_size)
            graph_reuse_max_meter.update(float(reuse_probs_tensor.max(dim=-1).values.mean()), batch_size)

            batch_edge_sum = edge_probs.mean(dim=1).sum(dim=0)
            batch_reuse_sum = reuse_probs_tensor.sum(dim=0)
            graph_edge_sum = batch_edge_sum if graph_edge_sum is None else graph_edge_sum + batch_edge_sum
            graph_reuse_sum = batch_reuse_sum if graph_reuse_sum is None else graph_reuse_sum + batch_reuse_sum
            graph_sample_count += batch_size

        if has_targets:
            loss_meter.update(float(losses["loss"].detach()), batch_size)
            final_loss_meter.update(float(losses["loss_final"].detach()), batch_size)
            aux_loss_meter.update(float(losses["loss_aux"].detach()), batch_size)
            final_targets.extend(batch["final"].detach().cpu().numpy().tolist())
            final_probs.extend(batch_probs_final.tolist())
            aux_batch_targets = batch["aux"].detach().cpu().numpy()
            aux_batch_masks = batch["aux_mask"].detach().cpu().numpy()
            for task_idx in range(len(base.AUX_COLUMNS)):
                valid = aux_batch_masks[:, task_idx] > 0
                if valid.any():
                    aux_targets[task_idx].extend(aux_batch_targets[valid, task_idx].tolist())
                    aux_probs[task_idx].extend(batch_probs_aux[valid, task_idx].tolist())

        image_ids = batch["image_id"] if isinstance(batch["image_id"], (list, tuple)) else [batch["image_id"]]
        seg_valid = batch["seg_valid"].detach().cpu().numpy()
        for idx, image_id in enumerate(image_ids):
            row: Dict[str, object] = {
                "Image": image_id,
                "epoch": int(epoch),
                "final_prob": float(batch_probs_final[idx]),
                "final_pred": int(batch_probs_final[idx] >= final_threshold),
                "seg_valid": float(seg_valid[idx]),
            }
            for stat_name in [
                "od_area_ratio",
                "oc_area_ratio",
                "cdr_area",
                "overlap_ratio",
                "peak_distance",
                "od_peak_border_margin",
                "oc_peak_border_margin",
                "od_class",
                "oc_class",
                "od_prob_mean",
                "oc_prob_mean",
                "od_prob_max",
                "oc_prob_max",
            ]:
                key = f"seg_{stat_name}"
                if key in batch:
                    row[key] = float(batch[key][idx].detach().cpu().item())
            for task_idx, task_name in enumerate(base.AUX_COLUMNS):
                row[f"{task_name}_prob"] = float(batch_probs_aux[idx, task_idx])
                row[f"{task_name}_pred"] = int(batch_probs_aux[idx, task_idx] >= 0.5)
            if has_targets:
                row["final_target"] = float(batch["final"][idx].detach().cpu().item())
                for task_idx, task_name in enumerate(base.AUX_COLUMNS):
                    row[f"{task_name}_target"] = float(batch["aux"][idx, task_idx].detach().cpu().item())
                    row[f"{task_name}_mask"] = float(batch["aux_mask"][idx, task_idx].detach().cpu().item())
            pred_rows.append(row)
        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}" if has_targets else f"samples={len(pred_rows)}",
            maps=f"{seg_valid_meter.avg:.3f}",
            q=f"{seg_q_mean_meter.avg:.3f}" if seg_q_mean_meter.count > 0 else "nan",
        )

    if has_targets:
        metrics: Dict[str, float] = {
            "val_loss": loss_meter.avg,
            "val_loss_final": final_loss_meter.avg,
            "val_loss_aux": aux_loss_meter.avg,
            "val_auroc_final": base.safe_auroc(final_targets, final_probs),
            "val_seg_valid_rate": seg_valid_meter.avg,
        }
        if seg_q_mean_meter.count > 0:
            metrics.update({
                "val_seg_q_mean": seg_q_mean_meter.avg,
                "val_seg_q_std": seg_q_std_meter.avg,
                "val_seg_q_min": seg_q_min_meter.avg,
                "val_seg_q_max": seg_q_max_meter.avg,
                "val_uncertainty_lambda": uncertainty_lambda_meter.avg,
            })
        aux_aurocs = []
        for name, targets, probs in zip(base.AUX_COLUMNS, aux_targets, aux_probs):
            auc = base.safe_auroc(targets, probs)
            metrics[f"val_auroc_{name}"] = auc
            if not math.isnan(auc):
                aux_aurocs.append(auc)
        metrics["val_auroc_aux_mean"] = float(np.mean(aux_aurocs)) if aux_aurocs else float("nan")
        if gate_entropy_meter.count > 0:
            metrics.update({
                "val_moe_gate_entropy": gate_entropy_meter.avg,
                "val_moe_gate_max": gate_max_meter.avg,
                "val_moe_task_expert_usage": task_expert_usage_meter.avg,
            })
        if graph_edge_entropy_meter.count > 0:
            metrics.update({
                "val_aux_graph_edge_entropy": graph_edge_entropy_meter.avg,
                "val_aux_graph_edge_max": graph_edge_max_meter.avg,
                "val_aux_graph_reuse_entropy": graph_reuse_entropy_meter.avg,
                "val_aux_graph_reuse_max": graph_reuse_max_meter.avg,
            })
    else:
        metrics = {"eval_samples": float(len(pred_rows)), "eval_seg_valid_rate": seg_valid_meter.avg}
        if seg_q_mean_meter.count > 0:
            metrics.update({
                "eval_seg_q_mean": seg_q_mean_meter.avg,
                "eval_seg_q_std": seg_q_std_meter.avg,
                "eval_seg_q_min": seg_q_min_meter.avg,
                "eval_seg_q_max": seg_q_max_meter.avg,
                "eval_uncertainty_lambda": uncertainty_lambda_meter.avg,
            })
        if gate_entropy_meter.count > 0:
            metrics.update({
                "eval_moe_gate_entropy": gate_entropy_meter.avg,
                "eval_moe_gate_max": gate_max_meter.avg,
                "eval_moe_task_expert_usage": task_expert_usage_meter.avg,
            })
        if graph_edge_entropy_meter.count > 0:
            metrics.update({
                "eval_aux_graph_edge_entropy": graph_edge_entropy_meter.avg,
                "eval_aux_graph_edge_max": graph_edge_max_meter.avg,
                "eval_aux_graph_reuse_entropy": graph_reuse_entropy_meter.avg,
                "eval_aux_graph_reuse_max": graph_reuse_max_meter.avg,
            })
    if graph_diagnostics_dir is not None and graph_sample_count > 0 and graph_edge_sum is not None and graph_reuse_sum is not None:
        save_aux_gat_diagnostics(
            edge_attention=(graph_edge_sum / float(graph_sample_count)).numpy(),
            node_reuse=(graph_reuse_sum / float(graph_sample_count)).numpy(),
            task_names=base.AUX_COLUMNS,
            output_dir=graph_diagnostics_dir,
            split_name=graph_split_name,
            epoch=epoch,
            graph_fixed=graph_fixed,
        )
    return metrics, pd.DataFrame(pred_rows) if pred_rows else None


def format_film_metrics(metrics: Dict[str, float]) -> str:
    preferred = [
        "epoch",
        "eval_samples",
        "eval_seg_valid_rate",
        "eval_loss",
        "eval_loss_final",
        "eval_loss_aux",
        "eval_auroc_final",
        "eval_auroc_aux_mean",
        "eval_seg_q_mean",
        "eval_seg_q_std",
        "eval_uncertainty_lambda",
        "eval_moe_gate_entropy",
        "eval_moe_gate_max",
        "eval_moe_task_expert_usage",
        "eval_aux_graph_edge_entropy",
        "eval_aux_graph_edge_max",
        "eval_aux_graph_reuse_entropy",
        "eval_aux_graph_reuse_max",
        "train_loss",
        "train_loss_final",
        "train_loss_aux",
        "train_seg_valid_rate",
        "train_seg_q_mean",
        "train_seg_q_std",
        "train_uncertainty_lambda",
        "train_moe_gate_loss",
        "train_moe_gate_entropy",
        "train_moe_gate_max",
        "train_moe_task_expert_usage",
        "val_loss",
        "val_loss_final",
        "val_loss_aux",
        "val_auroc_final",
        "val_auroc_aux_mean",
        "val_seg_valid_rate",
        "val_seg_q_mean",
        "val_seg_q_std",
        "val_uncertainty_lambda",
        "val_moe_gate_entropy",
        "val_moe_gate_max",
        "val_moe_task_expert_usage",
        "val_aux_graph_edge_entropy",
        "val_aux_graph_edge_max",
        "val_aux_graph_reuse_entropy",
        "val_aux_graph_reuse_max",
        "test_loss",
        "test_loss_final",
        "test_loss_aux",
        "test_auroc_final",
        "test_auroc_aux_mean",
        "test_seg_valid_rate",
        "test_seg_q_mean",
        "test_seg_q_std",
        "test_uncertainty_lambda",
        "test_aux_graph_edge_entropy",
        "test_aux_graph_edge_max",
        "test_aux_graph_reuse_entropy",
        "test_aux_graph_reuse_max",
        "lr_image",
        "lr_prior",
        "lr_head",
        "val_prob_mean_final",
        "val_prob_std_final",
        "val_pred_pos_rate_final",
    ]
    parts = []
    for key in preferred:
        if key in metrics:
            value = metrics[key]
            if math.isnan(value):
                parts.append(f"{key}=nan")
            elif key.startswith("lr_"):
                parts.append(f"{key}={value:.2e}")
            else:
                parts.append(f"{key}={value:.4f}")
    return " | ".join(parts)


def _forward_structural_heads_from_features(
    model: StructuralPriorModulationModel,
    image_features: torch.Tensor,
    geometry: Optional[torch.Tensor],
    geometry_mask: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    features = model.fuse_features(
        image_features=image_features,
        geometry=geometry,
        geometry_mask=geometry_mask,
    )
    if isinstance(getattr(model, "task_moe_head", None), TaskFeatureGatedMoEMTLHead):
        task_features = features.unsqueeze(1).expand(-1, model.task_moe_head.num_tasks, -1)
        return model.head_outputs_from_task_features(task_features)
    return model.head_outputs(features)


def compute_structural_prior_gradcam(
    model: StructuralPriorModulationModel,
    image: torch.Tensor,
    seg_input: torch.Tensor,
    seg_valid: Optional[torch.Tensor],
    geometry: Optional[torch.Tensor],
    geometry_mask: Optional[torch.Tensor],
    task_index: int,
    is_final_task: bool,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    if getattr(model, "group_residual_channel_film", None) is not None:
        spatial_by_task, _, _ = model.forward_shared_film_group_residual_backbone(
            image,
            seg_input,
            seg_valid=seg_valid,
        )
        task_feature_index = 0 if is_final_task else task_index + 1
        task_name = model.task_names[task_feature_index]
        spatial_for_cam = spatial_by_task[task_name]
        spatial_for_cam.retain_grad()
        task_image_features = torch.stack(
            [model.pool_image_features(spatial_by_task[name]) for name in model.task_names],
            dim=1,
        )
        task_features = torch.stack(
            [
                model.fuse_features(
                    image_features=task_image_features[:, feature_index],
                    geometry=geometry,
                    geometry_mask=geometry_mask,
                )
                for feature_index in range(task_image_features.shape[1])
            ],
            dim=1,
        )
        outputs = model.head_outputs_from_task_features(task_features)
        target_logit = outputs["final_logits"][0] if is_final_task else outputs["aux_logits"][0, task_index]
        target_logit.backward()

        gradients = spatial_for_cam.grad
        if gradients is None:
            raise RuntimeError("Gradients were not retained for SGR-TaFM Grad-CAM computation.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * spatial_for_cam).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)[0].detach().cpu().numpy()
        cam = cv2.resize(cam, dsize=(image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_LINEAR)
        cam = cam - cam.min()
        cam = cam / max(cam.max(), 1e-8)
        return cam

    if getattr(model, "aux_residual_channel_film", None) is not None:
        spatial_by_task, _, _ = model.forward_shared_film_aux_residual_backbone(
            image,
            seg_input,
            seg_valid=seg_valid,
        )
        task_feature_index = 0 if is_final_task else task_index + 1
        task_name = model.task_names[task_feature_index]
        spatial_for_cam = spatial_by_task[task_name]
        spatial_for_cam.retain_grad()
        task_image_features = torch.stack(
            [model.pool_image_features(spatial_by_task[name]) for name in model.task_names],
            dim=1,
        )
        task_features = torch.stack(
            [
                model.fuse_features(
                    image_features=task_image_features[:, feature_index],
                    geometry=geometry,
                    geometry_mask=geometry_mask,
                )
                for feature_index in range(task_image_features.shape[1])
            ],
            dim=1,
        )
        outputs = model.head_outputs_from_task_features(task_features)
        target_logit = outputs["final_logits"][0] if is_final_task else outputs["aux_logits"][0, task_index]
        target_logit.backward()

        gradients = spatial_for_cam.grad
        if gradients is None:
            raise RuntimeError("Gradients were not retained for shared-FiLM aux-residual Grad-CAM computation.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * spatial_for_cam).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)[0].detach().cpu().numpy()
        cam = cv2.resize(cam, dsize=(image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_LINEAR)
        cam = cam - cam.min()
        cam = cam / max(cam.max(), 1e-8)
        return cam

    if getattr(model, "task_channel_film", None) is not None:
        spatial_by_task, _, _ = model.forward_spatial_task_channel_backbone(
            image,
            seg_input,
            seg_valid=seg_valid,
        )
        task_feature_index = 0 if is_final_task else task_index + 1
        task_name = model.task_names[task_feature_index]
        spatial_for_cam = spatial_by_task[task_name]
        spatial_for_cam.retain_grad()
        task_image_features = torch.stack(
            [model.pool_image_features(spatial_by_task[name]) for name in model.task_names],
            dim=1,
        )
        task_features = torch.stack(
            [
                model.fuse_features(
                    image_features=task_image_features[:, feature_index],
                    geometry=geometry,
                    geometry_mask=geometry_mask,
                )
                for feature_index in range(task_image_features.shape[1])
            ],
            dim=1,
        )
        outputs = model.head_outputs_from_task_features(task_features)
        target_logit = outputs["final_logits"][0] if is_final_task else outputs["aux_logits"][0, task_index]
        target_logit.backward()

        gradients = spatial_for_cam.grad
        if gradients is None:
            raise RuntimeError("Gradients were not retained for spatial-task-channel FiLM Grad-CAM computation.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * spatial_for_cam).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)[0].detach().cpu().numpy()
        cam = cv2.resize(cam, dsize=(image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_LINEAR)
        cam = cam - cam.min()
        cam = cam / max(cam.max(), 1e-8)
        return cam

    if getattr(model, "task_aware_film", None) is not None:
        spatial_by_group, _, _ = model.forward_task_aware_backbone(
            image,
            seg_input,
            seg_valid=seg_valid,
        )
        task_feature_index = 0 if is_final_task else task_index + 1
        group_name = model.tafm_task_group_names[task_feature_index]
        spatial_for_cam = spatial_by_group[group_name]
        spatial_for_cam.retain_grad()
        pooled_by_group = {
            name: model.pool_image_features(spatial_features)
            for name, spatial_features in spatial_by_group.items()
        }
        task_image_features = torch.stack(
            [pooled_by_group[name] for name in model.tafm_task_group_names],
            dim=1,
        )
        task_features = torch.stack(
            [
                model.fuse_features(
                    image_features=task_image_features[:, feature_index],
                    geometry=geometry,
                    geometry_mask=geometry_mask,
                )
                for feature_index in range(task_image_features.shape[1])
            ],
            dim=1,
        )
        outputs = model.head_outputs_from_task_features(task_features)
        target_logit = outputs["final_logits"][0] if is_final_task else outputs["aux_logits"][0, task_index]
        target_logit.backward()

        gradients = spatial_for_cam.grad
        if gradients is None:
            raise RuntimeError("Gradients were not retained for task-aware FiLM Grad-CAM computation.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * spatial_for_cam).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)[0].detach().cpu().numpy()
        cam = cv2.resize(cam, dsize=(image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_LINEAR)
        cam = cam - cam.min()
        cam = cam / max(cam.max(), 1e-8)
        return cam

    spatial_for_cam, _, _ = model.forward_backbone_spatial(image, seg_input, seg_valid=seg_valid)
    spatial_for_cam.retain_grad()
    image_features = model.pool_image_features(spatial_for_cam)
    outputs = _forward_structural_heads_from_features(
        model=model,
        image_features=image_features,
        geometry=geometry,
        geometry_mask=geometry_mask,
    )
    target_logit = outputs["final_logits"][0] if is_final_task else outputs["aux_logits"][0, task_index]
    target_logit.backward()

    gradients = spatial_for_cam.grad
    if gradients is None:
        raise RuntimeError("Gradients were not retained for structural-prior Grad-CAM computation.")
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * spatial_for_cam).sum(dim=1, keepdim=False)
    cam = torch.relu(cam)[0].detach().cpu().numpy()
    cam = cv2.resize(cam, dsize=(image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_LINEAR)
    cam = cam - cam.min()
    cam = cam / max(cam.max(), 1e-8)
    return cam


def save_film_gradcam_visualizations(
    model: StructuralPriorModulationModel,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    split_name: str,
    max_samples: int,
    balance_buckets: bool = True,
) -> None:
    if max_samples <= 0:
        return

    gradcam_dir = output_dir / "gradcam" / split_name
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    print(f"Structural-prior Grad-CAM output dir -> {gradcam_dir}")

    model.eval()
    saved_by_bucket = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "unlabeled": 0}
    per_bucket_quota = max(1, math.ceil(max_samples / 4))
    label_map = {0: "N", 1: "T"}
    task_names = [("final", True, -1)] + [(task_name, False, task_idx) for task_idx, task_name in enumerate(base.AUX_COLUMNS)]

    for batch in loader:
        batch = base.move_batch_to_device(batch, device)
        image_ids = batch["image_id"] if isinstance(batch["image_id"], (list, tuple)) else [batch["image_id"]]

        for batch_index, image_id in enumerate(image_ids):
            labeled_total = sum(saved_by_bucket[name] for name in ["tp", "tn", "fp", "fn"])
            if labeled_total >= max_samples:
                print(f"Saved structural-prior Grad-CAM visualizations -> {gradcam_dir}")
                print(f"Structural-prior Grad-CAM bucket counts -> {saved_by_bucket}")
                return

            image_tensor = batch["image"][batch_index : batch_index + 1]
            seg_input = batch["seg_input"][batch_index : batch_index + 1]
            seg_valid = batch.get("seg_valid")
            seg_valid_sample = seg_valid[batch_index : batch_index + 1] if torch.is_tensor(seg_valid) else None
            geometry = batch.get("geometry")
            geometry_mask = batch.get("geometry_mask")
            geometry_sample = geometry[batch_index : batch_index + 1] if torch.is_tensor(geometry) else None
            geometry_mask_sample = geometry_mask[batch_index : batch_index + 1] if torch.is_tensor(geometry_mask) else None

            image_rgb = base.denormalize_image_tensor(image_tensor[0])
            panel_h, panel_w = image_rgb.shape[:2]

            with torch.no_grad():
                outputs = forward_film(
                    model,
                    {
                        "image": image_tensor,
                        "seg_input": seg_input,
                        "seg_valid": seg_valid_sample,
                        "geometry": geometry_sample,
                        "geometry_mask": geometry_mask_sample,
                    },
                )
                final_prob = float(torch.sigmoid(outputs["final_logits"])[0].detach().cpu().item())
                aux_probs = torch.sigmoid(outputs["aux_logits"])[0].detach().cpu().numpy()
                structural_prior = outputs["structural_prior"][0].detach().float().cpu().numpy()

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

            if (
                balance_buckets
                and outcome_bucket != "unlabeled"
                and saved_by_bucket[outcome_bucket] >= per_bucket_quota
            ):
                continue
            if (
                balance_buckets
                and outcome_bucket == "unlabeled"
                and saved_by_bucket[outcome_bucket] >= max(1, min(2, max_samples))
            ):
                continue

            sample_dir = gradcam_dir / outcome_bucket
            sample_dir.mkdir(parents=True, exist_ok=True)
            pred_letter = label_map.get(final_pred, final_pred)
            base_title = f"orig final p={final_prob:.2f} pred={pred_letter}"
            if final_target is not None:
                base_title += f" tgt={label_map.get(final_target, final_target)}"
            seg_title = "QC disabled"
            prior_names = structural_prior_channel_names(model.structural_prior_mode)
            task_panels: List[np.ndarray] = [base.annotate_panel(image_rgb, base_title)]
            for prior_index, prior_name in enumerate(prior_names):
                prior_rgb, prior_mean, prior_max = _render_softmap_rgb(
                    structural_prior[prior_index],
                    panel_h,
                    panel_w,
                )
                title_suffix = f" {seg_title}" if prior_index == 0 else ""
                task_panels.append(
                    base.annotate_panel(
                        prior_rgb,
                        f"{prior_name} soft{title_suffix} max={prior_max:.3f} mean={prior_mean:.3f}",
                    )
                )

            for task_name, is_final_task, task_index in task_names:
                cam_map = compute_structural_prior_gradcam(
                    model=model,
                    image=image_tensor,
                    seg_input=seg_input,
                    seg_valid=seg_valid_sample,
                    geometry=geometry_sample,
                    geometry_mask=geometry_mask_sample,
                    task_index=task_index,
                    is_final_task=is_final_task,
                )
                overlay = base.render_gradcam_overlay(image_rgb, cam_map)
                if is_final_task:
                    task_prob = final_prob
                    task_pred = final_pred
                    task_target = final_target
                else:
                    task_prob = float(aux_probs[task_index])
                    task_pred = int(task_prob >= 0.5)
                    task_target = int(float(batch["aux"][batch_index, task_index].detach().cpu().item()) >= 0.5) if "aux" in batch else None
                title = f"{task_name} modulated p={task_prob:.2f} pred={label_map.get(task_pred, task_pred)}"
                if task_target is not None:
                    title += f" tgt={label_map.get(task_target, task_target)}"
                task_panels.append(base.annotate_panel(overlay, title))

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
            grid_output_path = sample_dir / f"{image_id}_structural_prior_grid.jpg"
            grid_saved = cv2.imwrite(str(grid_output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            if not grid_saved:
                print(f"Warning: failed to save structural-prior Grad-CAM grid -> {grid_output_path}")
            elif sum(saved_by_bucket.values()) < 5:
                print(f"Saved structural-prior Grad-CAM grid -> {grid_output_path}")

            saved_by_bucket[outcome_bucket] += 1

    print(f"Saved structural-prior Grad-CAM visualizations -> {gradcam_dir}")
    print(f"Structural-prior Grad-CAM bucket counts -> {saved_by_bucket}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structural-prior modulation for referral glaucoma detection.")
    parser.add_argument("--csv", type=str, default=base.DEFAULT_CSV_PATH)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--eval-csv", "--test-csv", dest="eval_csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=base.DEFAULT_IMAGE_DIR)
    parser.add_argument("--cache-dir", type=str, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-images-name", type=str, default="images.npy")
    parser.add_argument("--cache-paths-name", type=str, default="paths.npy")
    parser.add_argument("--cache-lookup", type=str, default="path", choices=["path", "index"])
    parser.add_argument("--allow-unlabeled-eval", action="store_true")
    parser.add_argument("--disable-npy-cache", action="store_true")
    parser.add_argument("--disable-cache-mmap", action="store_true")
    parser.add_argument("--disable-cache-index-fallback", action="store_true")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_FILM_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=str(Path(DEFAULT_FILM_OUTPUT_DIR) / "best.pt"))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--preds-csv", type=str, default=None)
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--warm-start-checkpoint", type=str, default=None)
    parser.add_argument("--stage", type=str, default="stage1", choices=list(base.STAGE_TASKS.keys()))
    parser.add_argument("--aux-tasks", type=str, default=None)
    parser.add_argument("--aux-loss-tasks", type=str, default="all")
    parser.add_argument("--run-stage2-after-stage1", dest="run_stage2_after_stage1", action="store_true")
    parser.add_argument("--disable-run-stage2-after-stage1", dest="run_stage2_after_stage1", action="store_false")
    parser.add_argument("--image-model-name", type=str, default="convnext_tiny")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--disable-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image-lr", type=float, default=3e-5)
    parser.add_argument("--prior-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fusion-dim", type=int, default=512)
    parser.add_argument("--prior-hidden-dim", type=int, default=DEFAULT_PRIOR_HIDDEN_DIM)
    parser.add_argument("--prior-dropout", type=float, default=DEFAULT_PRIOR_DROPOUT)
    parser.add_argument("--prior-dilation-kernel", type=int, default=DEFAULT_PRIOR_DILATION_KERNEL)
    parser.add_argument("--use-structural-prior", dest="use_structural_prior", action="store_true")
    parser.add_argument("--disable-structural-prior", dest="use_structural_prior", action="store_false")
    parser.add_argument("--softmap-input-mode", type=str, default="od_oc", choices=["od_oc", "od_oc_peri"])
    parser.add_argument("--structural-prior-mode", type=str, default="derived4", choices=["derived4", "derived6", "explicit3"])
    parser.add_argument(
        "--structural-prior-integration",
        type=str,
        default="film",
        choices=[
            "film",
            "input_concat",
            "cross_attention",
            "task_aware_film",
            "task_aware_uncertainty_film",
            "spatial_task_channel_film",
            "spatial_task_channel_uncertainty_film",
            "shared_film_aux_residual",
            "shared_film_group_residual",
        ],
    )
    parser.add_argument("--tafm-grouping", type=str, default="clinical", choices=["clinical", "task"])
    parser.add_argument("--prior-attention-dim", type=int, default=192)
    parser.add_argument("--prior-attention-heads", type=int, default=4)
    parser.add_argument("--prior-attention-stage", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--head-type",
        type=str,
        default="simple",
        choices=["simple", "moe_mtl", "aux_reuse_moe", "aux_gat_reuse_moe", "aux_gat_reuse_mtl", "aux_gcn_reuse_mtl", "grouped_moe", "hybrid_grouped_moe"],
    )
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--moe-dim", type=int, default=256)
    parser.add_argument("--moe-hidden-dim", type=int, default=512)
    parser.add_argument("--task-tower-dim", type=int, default=128)
    parser.add_argument("--moe-balance-loss-weight", type=float, default=0.01)
    parser.add_argument("--moe-entropy-loss-weight", type=float, default=0.001)
    parser.add_argument("--aux-delta-initial-scale", type=float, default=0.01)
    parser.add_argument("--geometry-csv", type=str, default=DEFAULT_GEOMETRY_CSV)
    parser.add_argument(
        "--geometry-features",
        type=str,
        default=",".join(DEFAULT_GEOMETRY_FEATURE_COLUMNS),
        help="Comma-separated geometry feature columns to use from --geometry-csv.",
    )
    parser.add_argument("--geometry-hidden-dim", type=int, default=128)
    parser.add_argument("--geometry-normalize", dest="geometry_normalize", action="store_true")
    parser.add_argument("--disable-geometry-normalize", dest="geometry_normalize", action="store_false")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-gradcam", dest="save_gradcam", action="store_true")
    parser.add_argument("--disable-save-gradcam", dest="save_gradcam", action="store_false")
    parser.add_argument("--gradcam-samples", type=int, default=20)
    parser.add_argument("--gradcam-split", type=str, default="test", choices=["test", "eval"])
    parser.add_argument("--threshold-mode", type=str, default="sensitivity", choices=["sensitivity", "youden", "f1", "fixed"])
    parser.add_argument("--val-threshold-mode", type=str, default="youden", choices=["sensitivity", "youden", "f1", "fixed"])
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--val-target-sensitivity", type=float, default=None)
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--val-fixed-threshold", type=float, default=None)
    parser.add_argument("--final-pos-weight-max", type=float, default=10.0)
    parser.add_argument("--aux-pos-weight-max", type=float, default=10.0)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--balanced-final-sampler", action="store_true")
    parser.add_argument("--disable-balanced-final-sampler", action="store_true")
    parser.add_argument("--disable-prior-bias-init", action="store_true")
    parser.add_argument("--seg-checkpoint", type=str, default=DEFAULT_SEG_CHECKPOINT)
    parser.add_argument("--seg-cache-dir", type=str, default=DEFAULT_CROP224_SEG_CACHE_DIR)
    parser.add_argument("--seg-memmap-name", type=str, default=DEFAULT_SEG_MEMMAP_NAME)
    parser.add_argument("--seg-memmap-ids-name", type=str, default=DEFAULT_SEG_MEMMAP_IDS_NAME)
    parser.add_argument("--rebuild-seg-memmap", action="store_true")
    parser.add_argument("--build-seg-cache-only", action="store_true", help="Build Swin OD/OC cache plus consolidated memmap, then exit.")
    parser.add_argument("--overwrite-seg-cache", action="store_true")
    parser.add_argument("--seg-infer-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--channels-last", dest="channels_last", action="store_true")
    parser.add_argument("--disable-channels-last", dest="channels_last", action="store_false")
    parser.set_defaults(
        pretrained=True,
        geometry_normalize=True,
        save_gradcam=True,
        run_stage2_after_stage1=True,
        channels_last=False,
        use_structural_prior=True,
    )
    args = parser.parse_args(argv)
    if args.lr is not None:
        args.image_lr = args.lr
        args.prior_lr = args.lr
        args.head_lr = args.lr
    args.model_name = DEFAULT_SEGMENTATION_MODEL_NAME
    return args


def run_single_stage(args: argparse.Namespace) -> None:
    if args.stage == "stage1" and args.head_type != "simple" and not args.eval_only:
        print("Stage1 policy override -> using simple MTL head for representation warm-up.")
        args.head_type = "simple"
    if args.stage == "stage2" and args.head_type == "simple" and not args.eval_only:
        print("Stage2 policy override -> using compact MoE-MTL head for rare auxiliary tasks.")
        for key, value in STAGE2_SMALL_MOE_CONFIG.items():
            setattr(args, key, value)

    selected_aux_tasks = base.resolve_selected_aux_tasks(args)
    base.configure_aux_tasks(selected_aux_tasks)
    aux_loss_setting = str(getattr(args, "aux_loss_tasks", "all") or "all").strip()
    if aux_loss_setting.lower() == "all":
        aux_loss_task_names = list(base.AUX_COLUMNS)
    else:
        aux_loss_task_names = [name.strip() for name in aux_loss_setting.split(",") if name.strip()]
    aux_loss_task_indices = resolve_aux_loss_task_indices(aux_loss_task_names)
    args.aux_loss_task_names = aux_loss_task_names
    args.geometry_feature_columns = [
        column.strip() for column in args.geometry_features.split(",") if column.strip()
    ] if args.geometry_csv else []
    args.geometry_norm_stats = {}
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    output_dir = (
        Path(args.output_dir)
        if getattr(args, "flat_output_dir", False)
        else film_stage_output_dir(args.output_dir, args.stage)
    )

    model = StructuralPriorModulationModel(
        image_model_name=args.image_model_name,
        pretrained=args.pretrained,
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
        structural_prior_mode=args.structural_prior_mode,
        use_structural_prior=getattr(args, "use_structural_prior", True),
        structural_prior_integration=getattr(args, "structural_prior_integration", "film"),
        tafm_grouping=getattr(args, "tafm_grouping", "clinical"),
        prior_attention_dim=getattr(args, "prior_attention_dim", 192),
        prior_attention_heads=getattr(args, "prior_attention_heads", 4),
        prior_attention_stage=getattr(args, "prior_attention_stage", 3),
        aux_delta_initial_scale=getattr(args, "aux_delta_initial_scale", 0.01),
    ).to(device)
    model.use_channels_last = bool(args.channels_last and device.type == "cuda")
    if model.use_channels_last:
        model.to(memory_format=torch.channels_last)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    optimizer = build_film_optimizer(model, args.image_lr, args.prior_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if args.eval_only:
        eval_loader = build_film_eval_loader(args)
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        load_structural_prior_checkpoint(checkpoint_path, model=model, optimizer=None, scheduler=None, scaler=None, device=device)
        graph_diagnostics_dir = output_dir / "gat_diagnostics" if args.head_type in {"aux_gat_reuse_moe", "aux_gat_reuse_mtl", "aux_gcn_reuse_mtl"} else None
        metrics, predictions = validate_film(
            model,
            eval_loader,
            device,
            epoch=0,
            amp_enabled=amp_enabled,
            aux_loss_task_indices=aux_loss_task_indices,
            graph_diagnostics_dir=graph_diagnostics_dir,
            graph_split_name=args.gradcam_split,
        )
        if predictions is not None and "final_target" in predictions:
            eval_threshold = find_best_threshold(
                predictions["final_target"],
                predictions["final_prob"],
                mode=args.threshold_mode,
                target_sensitivity=args.target_sensitivity,
                fixed_threshold=args.fixed_threshold,
            )
            metrics, predictions = validate_film(
                model,
                eval_loader,
                device,
                epoch=0,
                amp_enabled=amp_enabled,
                final_threshold=eval_threshold,
                aux_loss_task_indices=aux_loss_task_indices,
                graph_diagnostics_dir=graph_diagnostics_dir,
                graph_split_name=args.gradcam_split,
            )
            print(
                f"Eval threshold by {args.threshold_mode}"
                + (f"(target_sensitivity={args.target_sensitivity:.3f})" if args.threshold_mode == "sensitivity" else "")
                + f" -> {threshold_summary(predictions['final_target'], predictions['final_prob'], eval_threshold)}"
            )
        metrics = base.rename_metric_prefix(metrics, "val_", "eval_")
        print(f"Eval checkpoint: {checkpoint_path}")
        print(format_film_metrics(metrics))
        if predictions is not None:
            saved_cm_path = base.save_multitask_confusion_matrices(predictions, output_dir, split_name=args.gradcam_split, epoch=None)
            if saved_cm_path is not None:
                print(f"Saved eval multitask confusion matrix to {saved_cm_path}")
        if predictions is not None and args.preds_csv:
            base.save_predictions_csv(predictions, Path(args.preds_csv))
            print(f"Saved predictions to {args.preds_csv}")
        if args.save_gradcam:
            save_film_gradcam_visualizations(
                model=model,
                loader=eval_loader,
                device=device,
                output_dir=output_dir,
                split_name=args.gradcam_split,
                max_samples=args.gradcam_samples,
            )
        return

    train_loader, val_loader, test_loader, final_pos_weight, aux_pos_weight, final_prior, aux_priors = build_film_loaders(args)
    apply_train_npmi_adjacency_if_available(model, train_loader)
    final_pos_weight = final_pos_weight.to(device)
    aux_pos_weight = aux_pos_weight.to(device)
    if not args.resume and not args.disable_prior_bias_init:
        initialize_film_output_biases(model, final_prior=final_prior, aux_priors=aux_priors)
        print("Initialized output head biases from train-split positive priors.")

    integration = getattr(args, "structural_prior_integration", "film")
    if not getattr(args, "use_structural_prior", True):
        modulation_description = "disabled"
    elif integration == "spatial_task_channel_film":
        modulation_description = "shared-spatial@stage2+task-channel@stage3"
    elif integration == "spatial_task_channel_uncertainty_film":
        modulation_description = "shared-spatial@stage2+qseg-residual@stage2+task-channel@stage3"
    elif integration == "task_aware_film":
        modulation_description = f"task-aware-FiLM@stage{getattr(args, 'prior_attention_stage', 2)}"
    elif integration == "task_aware_uncertainty_film":
        modulation_description = f"task-aware-FiLM@stage{getattr(args, 'prior_attention_stage', 2)}+qseg-residual"
    elif integration == "shared_film_aux_residual":
        modulation_description = "shared-FiLM@stages2.3+aux-residual-channel-FiLM@stage3"
    elif integration == "shared_film_group_residual":
        modulation_description = "shared-FiLM@stages2.3+group-residual-TaFM@stage3"
    elif integration == "film":
        modulation_description = "spatial-channel-FiLM@stages2.3"
    else:
        modulation_description = "disabled"

    print(
        "Training setup -> "
        f"image_encoder={_resolve_model_name(args.image_model_name)} | "
        f"structural_prior={structural_prior_description(args.structural_prior_mode) if getattr(args, 'use_structural_prior', True) else 'disabled'} | "
        f"integration={getattr(args, 'structural_prior_integration', 'film') if getattr(args, 'use_structural_prior', True) else 'none'} | "
        f"tafm_grouping={getattr(args, 'tafm_grouping', 'clinical') if getattr(args, 'structural_prior_integration', 'film') in {'task_aware_film', 'task_aware_uncertainty_film', 'shared_film_group_residual'} else 'disabled'} | "
        f"cross_attention_stage={getattr(args, 'prior_attention_stage', 3) if getattr(args, 'structural_prior_integration', 'film') == 'cross_attention' else 'disabled'} | "
        f"modulation={modulation_description} | "
        f"prior_dropout={args.prior_dropout:.2f} | fusion={'RGB+geometry' if args.geometry_feature_columns else 'RGB'}->{args.fusion_dim} | head={args.head_type} | "
        f"moe_shared_experts={args.num_experts if args.head_type in {'moe_mtl', 'aux_reuse_moe', 'aux_gat_reuse_moe', 'grouped_moe', 'hybrid_grouped_moe'} else 0} | "
        f"moe_balance={args.moe_balance_loss_weight if args.head_type in {'moe_mtl', 'aux_reuse_moe', 'aux_gat_reuse_moe', 'aux_gat_reuse_mtl', 'aux_gcn_reuse_mtl', 'grouped_moe', 'hybrid_grouped_moe'} else 0.0:g} | "
        f"moe_entropy={args.moe_entropy_loss_weight if args.head_type in {'moe_mtl', 'aux_reuse_moe', 'aux_gat_reuse_moe', 'aux_gat_reuse_mtl', 'aux_gcn_reuse_mtl', 'grouped_moe', 'hybrid_grouped_moe'} else 0.0:g} | "
        f"stage={args.stage} | "
        f"aux_tasks={base.AUX_COLUMNS if base.AUX_COLUMNS else ['final_only']} | "
        f"aux_loss_tasks={aux_loss_task_names} | "
        f"image_preprocessing=precomputed_crop224 | "
        f"geometry={'on' if args.geometry_feature_columns else 'off'} | "
        f"geometry_dim={len(args.geometry_feature_columns)} | "
        f"geometry_normalize={args.geometry_normalize} | "
        f"seg_memmap={Path(args.seg_cache_dir) / args.seg_memmap_name} | "
        f"qc=disabled | workers={args.num_workers} | prefetch={args.prefetch_factor}"
    )

    start_epoch = 1
    best_val_auroc = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    metrics_history: List[Dict[str, float]] = []
    metrics_csv_path = Path(args.metrics_csv) if args.metrics_csv else output_dir / "metrics_history.csv"
    if args.resume:
        start_epoch, best_val_auroc = load_structural_prior_checkpoint(
            Path(args.resume), model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device
        )
        completed_epoch = start_epoch - 1
        best_epoch = completed_epoch
        if metrics_csv_path.exists():
            previous_metrics = pd.read_csv(metrics_csv_path)
            if "epoch" in previous_metrics.columns:
                previous_metrics = previous_metrics[previous_metrics["epoch"] <= completed_epoch]
            metrics_history = previous_metrics.to_dict(orient="records")
            print(f"Restored metrics history -> rows={len(metrics_history)} from {metrics_csv_path}")

        loaded_t_max = int(getattr(scheduler, "T_max", args.epochs))
        if args.epochs > loaded_t_max:
            remaining_epochs = max(1, args.epochs - completed_epoch)
            restart_lrs = [args.image_lr, args.prior_lr] + [args.head_lr] * max(0, len(optimizer.param_groups) - 2)
            for parameter_group, learning_rate in zip(optimizer.param_groups, restart_lrs):
                parameter_group["lr"] = learning_rate
                parameter_group["initial_lr"] = learning_rate
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs,
            )
            print(
                "Extended training schedule -> "
                f"completed={completed_epoch}, target={args.epochs}, remaining={remaining_epochs}, "
                f"lr_image={args.image_lr:.2e}, lr_prior={args.prior_lr:.2e}, lr_head={args.head_lr:.2e}"
            )
    elif args.warm_start_checkpoint:
        warm_start_path = Path(args.warm_start_checkpoint)
        if warm_start_path.exists():
            warm_start_matching_checkpoint(warm_start_path, model=model, device=device)
        else:
            print(f"Warm-start checkpoint not found, skipping: {warm_start_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch_film(
            model, train_loader, optimizer, scaler, device, epoch, amp_enabled,
            args.grad_clip_norm, final_pos_weight, aux_pos_weight,
            aux_loss_task_indices=aux_loss_task_indices,
            moe_balance_loss_weight=args.moe_balance_loss_weight if args.head_type in {"moe_mtl", "aux_reuse_moe", "aux_gat_reuse_moe", "aux_gat_reuse_mtl", "aux_gcn_reuse_mtl", "grouped_moe", "hybrid_grouped_moe"} else 0.0,
            moe_entropy_loss_weight=args.moe_entropy_loss_weight if args.head_type in {"moe_mtl", "aux_reuse_moe", "aux_gat_reuse_moe", "aux_gat_reuse_mtl", "aux_gcn_reuse_mtl", "grouped_moe", "hybrid_grouped_moe"} else 0.0,
        )
        val_metrics, val_predictions = validate_film(
            model, val_loader, device, epoch, amp_enabled,
            final_pos_weight=final_pos_weight, aux_pos_weight=aux_pos_weight,
            aux_loss_task_indices=aux_loss_task_indices,
            graph_diagnostics_dir=output_dir / "gat_diagnostics" if args.head_type in {"aux_gat_reuse_moe", "aux_gat_reuse_mtl", "aux_gcn_reuse_mtl"} else None,
            graph_split_name="val",
        )
        best_threshold = 0.5
        if val_predictions is not None and "final_target" in val_predictions:
            val_target_sensitivity = args.val_target_sensitivity if args.val_target_sensitivity is not None else args.target_sensitivity
            val_fixed_threshold = args.val_fixed_threshold if args.val_fixed_threshold is not None else args.fixed_threshold
            best_threshold = find_best_threshold(
                val_predictions["final_target"],
                val_predictions["final_prob"],
                mode=args.val_threshold_mode,
                target_sensitivity=val_target_sensitivity,
                fixed_threshold=val_fixed_threshold,
            )
            val_predictions["final_pred"] = (val_predictions["final_prob"].astype(float) >= best_threshold).astype(int)
            print(
                f"Validation threshold by {args.val_threshold_mode}"
                + (f"(target_sensitivity={val_target_sensitivity:.3f})" if args.val_threshold_mode == "sensitivity" else "")
                + f" -> {threshold_summary(val_predictions['final_target'], val_predictions['final_prob'], best_threshold)}"
            )
            saved_cm_path = base.save_multitask_confusion_matrices(val_predictions, output_dir, split_name="val", epoch=epoch)
            if saved_cm_path is not None:
                print(f"Saved validation multitask confusion matrix to {saved_cm_path}")

        scheduler.step()
        metrics = {
            **train_metrics,
            **val_metrics,
            **base.collect_probability_summary(val_predictions),
            "epoch": float(epoch),
            "lr_image": base._as_float(optimizer.param_groups[0]["lr"]),
            "lr_prior": base._as_float(optimizer.param_groups[1]["lr"]),
            "lr_head": base._as_float(optimizer.param_groups[2]["lr"]),
        }
        metrics_history.append(metrics.copy())
        base.save_metrics_history(metrics_history, metrics_csv_path)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | {format_film_metrics(metrics)}")

        current_auroc = metrics["val_auroc_final"]
        improved = not math.isnan(current_auroc) and current_auroc > (best_val_auroc + args.early_stopping_min_delta)
        if improved:
            best_val_auroc = current_auroc
            best_epoch = epoch
            epochs_without_improvement = 0
            base.save_checkpoint(output_dir, "best.pt", model, optimizer, scheduler, scaler, epoch, metrics, best_val_auroc, args)
            print(f"New best checkpoint at epoch {epoch:03d} | val_auroc_final={best_val_auroc:.4f}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0:
                print(f"No validation AUROC improvement for {epochs_without_improvement} epoch(s). Best epoch={best_epoch:03d}, best_val_auroc_final={best_val_auroc:.4f}")

        base.save_checkpoint(output_dir, "last.pt", model, optimizer, scheduler, scaler, epoch, metrics, best_val_auroc, args)
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch:03d}. Best epoch={best_epoch:03d}, best_val_auroc_final={best_val_auroc:.4f}")
            break

    final_checkpoint_path = output_dir / "best.pt" if (output_dir / "best.pt").exists() else output_dir / "last.pt"
    if final_checkpoint_path.exists():
        load_structural_prior_checkpoint(final_checkpoint_path, model=model, optimizer=None, scheduler=None, scaler=None, device=device)
        _, val_predictions_for_thr = validate_film(
            model, val_loader, device, epoch=0, amp_enabled=amp_enabled,
            final_pos_weight=final_pos_weight, aux_pos_weight=aux_pos_weight,
            aux_loss_task_indices=aux_loss_task_indices,
        )
        best_threshold = 0.5
        if val_predictions_for_thr is not None:
            best_threshold = find_best_threshold(
                val_predictions_for_thr["final_target"],
                val_predictions_for_thr["final_prob"],
                mode=args.threshold_mode,
                target_sensitivity=args.target_sensitivity,
                fixed_threshold=args.fixed_threshold,
            )
            print(
                f"Fixed threshold from validation set by {args.threshold_mode}"
                + (f"(target_sensitivity={args.target_sensitivity:.3f})" if args.threshold_mode == "sensitivity" else "")
                + f" -> {threshold_summary(val_predictions_for_thr['final_target'], val_predictions_for_thr['final_prob'], best_threshold)}"
            )
        else:
            print(f"Fixed threshold = {best_threshold:.4f}")
        test_metrics, test_predictions = validate_film(
            model, test_loader, device, epoch=0, amp_enabled=amp_enabled,
            final_pos_weight=final_pos_weight, aux_pos_weight=aux_pos_weight,
            final_threshold=best_threshold,
            aux_loss_task_indices=aux_loss_task_indices,
            graph_diagnostics_dir=output_dir / "gat_diagnostics" if args.head_type in {"aux_gat_reuse_moe", "aux_gat_reuse_mtl", "aux_gcn_reuse_mtl"} else None,
            graph_split_name="test",
        )
        test_metrics = base.rename_metric_prefix(test_metrics, "val_", "test_")
        print(f"Test checkpoint: {final_checkpoint_path}")
        print(format_film_metrics(test_metrics))
        if test_predictions is not None:
            saved_cm_path = base.save_multitask_confusion_matrices(test_predictions, output_dir, split_name="test", epoch=None)
            if saved_cm_path is not None:
                print(f"Saved test multitask confusion matrix to {saved_cm_path}")
        if test_predictions is not None and args.preds_csv:
            base.save_predictions_csv(test_predictions, Path(args.preds_csv))
            print(f"Saved predictions to {args.preds_csv}")
        if args.save_gradcam:
            save_film_gradcam_visualizations(
                model=model,
                loader=test_loader,
                device=device,
                output_dir=output_dir,
                split_name="test",
                max_samples=args.gradcam_samples,
            )


def main() -> None:
    args = parse_args()
    if args.build_seg_cache_only:
        base.seed_everything(args.seed)
        train_df, val_df, test_df = base.build_train_val_test_dataframes(args)
        all_df = pd.concat([train_df, val_df, test_df], axis=0)
        seg_softmap_cache = prepare_segmentation_memmap(args, all_df)
        print(f"Segmentation cache preparation complete -> entries={len(seg_softmap_cache.lookup)}")
        return
    if args.eval_only:
        checkpoint_aux_columns = base.read_checkpoint_aux_columns(Path(args.checkpoint))
        if checkpoint_aux_columns:
            args.aux_tasks = ",".join(checkpoint_aux_columns)
            print(f"Eval-only checkpoint aux task override -> {checkpoint_aux_columns}")
    base.seed_everything(args.seed)
    run_single_stage(args)

    if args.run_stage2_after_stage1 and args.stage == "stage1":
        stage2_args = copy.deepcopy(args)
        stage2_args.stage = "stage2"
        stage2_args.aux_tasks = None
        stage2_args.resume = None
        stage2_args.eval_only = False
        stage2_args.checkpoint = str(film_stage_output_dir(args.output_dir, "stage2") / "best.pt")
        stage2_args.warm_start_checkpoint = str(film_stage_output_dir(args.output_dir, "stage1") / "best.pt")
        for key, value in STAGE2_SMALL_MOE_CONFIG.items():
            setattr(stage2_args, key, value)
        print("\n" + "=" * 80)
        print("Stage1 complete. Starting automatic Stage2 training with small MoE-MTL head...")
        print(
            "Stage2 MoE config -> "
            f"num_experts={stage2_args.num_experts} | "
            f"moe_dim={stage2_args.moe_dim} | "
            f"moe_hidden_dim={stage2_args.moe_hidden_dim} | "
            f"task_tower_dim={stage2_args.task_tower_dim} | "
            f"image_lr={stage2_args.image_lr:.2e} | "
            f"prior_lr={stage2_args.prior_lr:.2e} | "
            f"head_lr={stage2_args.head_lr:.2e}"
        )
        print(f"Stage2 warm start checkpoint -> {stage2_args.warm_start_checkpoint}")
        print("=" * 80)
        run_single_stage(stage2_args)


if __name__ == "__main__":
    main()
