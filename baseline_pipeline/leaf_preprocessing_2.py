#!/usr/bin/env python3
"""
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen

leaf_preprocessing.py

Purpose:
    Build a clean baseline classification dataset from annotated COCO leaf data.

What this script does:
    1. Loads COCO annotations from train and val folders
    2. Finds all leaf instances in each image
    3. Selects the largest leaf instance by area
    4. Produces two crop variants:
         - bbox crop: rectangular crop around the largest leaf
         - masked crop: same crop, but background outside the leaf mask is removed
    5. Saves crops into an organized output directory
    6. Writes a metadata CSV for downstream ResNet training

Expected input structure:
    ./datasets/agai_correct/v3/coco/train/_annotations.coco.json
    ./datasets/agai_correct/v3/coco/val/_annotations.coco.json
    ./datasets/agai_correct/v3/coco/train/<images>
    ./datasets/agai_correct/v3/coco/val/<images>
    ./datasets/agai_correct/v3/tracker.csv   (optional but recommended)

Example output:
    ./datasets/processed_leaf_crops/
        bbox/
            train/
            val/
        masked/
            train/
            val/
        leaf_crop_metadata.csv

Notes:
    - This script assumes COCO instance segmentation annotations.
    - It selects the largest LEAF instance only.
    - It does NOT trust sample_ID as a true global tree ID.
      It preserves it as metadata only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw


# -----------------------------------------------------------------------------
# Data classes for cleaner organization
# -----------------------------------------------------------------------------

@dataclass
class ImageRecord:
    """
    Represents one image entry from the COCO file.
    """
    image_id: int
    file_name: str
    width: int
    height: int
    extra_name: Optional[str]
    split: str
    image_path: Path


@dataclass
class AnnotationRecord:
    """
    Represents one COCO annotation instance.
    """
    annotation_id: int
    image_id: int
    category_id: int
    bbox: List[float]
    area: float
    segmentation: Any
    iscrowd: int


@dataclass
class LargestLeafResult:
    """
    Stores the selected largest leaf annotation for an image.
    """
    annotation: AnnotationRecord
    mask: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_json(json_path: Path) -> dict:
    """
    Load a JSON file from disk.
    """
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    """
    Create a directory if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def parse_filename_metadata(file_name: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Parse ppm bin and sample ID from filenames like:
        ICP_600ppm_4_jpg.rf.7e7383a82fd37a49e34669e30683eff3.jpg
        ICP_1Kppm_2_jpg.rf.ca5aface4646e59540bd497e1ce176fb.jpg

    Returns:
        ppm_bin: int or None
        sample_id: int or None
        base_stub: str or None

    Notes:
        - 1Kppm is converted to 1000
        - sample_id is preserved as metadata only
    """
    # Remove the Roboflow hash suffix mentally by matching only the front part
    # before "_jpg.rf..."
    #
    # Example front part:
    #   ICP_600ppm_4
    #   ICP_1Kppm_2
    pattern = r"^(ICP)_(400ppm|600ppm|800ppm|1Kppm)_(\d+)"
    match = re.match(pattern, file_name)

    if not match:
        return None, None, None

    _, ppm_raw, sample_raw = match.groups()

    if ppm_raw == "1Kppm":
        ppm_bin = 1000
    else:
        ppm_bin = int(ppm_raw.replace("ppm", ""))

    sample_id = int(sample_raw)
    base_stub = f"ICP_{ppm_raw}_{sample_id}"

    return ppm_bin, sample_id, base_stub


def coco_bbox_to_xyxy(bbox: List[float], image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox

    if w < 0 or h < 0:
        print(f"[WARNING] Malformed bbox (negative dimension) — normalizing: {bbox}")

    raw_x1 = int(np.floor(x))
    raw_y1 = int(np.floor(y))
    raw_x2 = int(np.ceil(x + w))
    raw_y2 = int(np.ceil(y + h))

    x1 = max(0, min(raw_x1, raw_x2))
    y1 = max(0, min(raw_y1, raw_y2))
    x2 = min(image_width,  max(raw_x1, raw_x2))
    y2 = min(image_height, max(raw_y1, raw_y2))

    return x1, y1, x2, y2


def segmentation_to_mask(
    segmentation: Any,
    image_width: int,
    image_height: int
) -> np.ndarray:
    """
    Convert a COCO polygon segmentation into a binary mask.

    Supports polygon format:
        segmentation = [[x1, y1, x2, y2, ..., xn, yn], [...], ...]

    Returns:
        mask: np.ndarray of shape (H, W), dtype=uint8
              values are 0 or 1

    """
    mask_img = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask_img)

    if not isinstance(segmentation, list):
        raise ValueError("Unsupported segmentation format. Expected polygon list.")

    for polygon in segmentation:
        if not polygon:
            continue

        # COCO polygon is a flat list: [x1, y1, x2, y2, ...]
        if len(polygon) < 6:
            # Too few points to form a polygon
            continue

        xy = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        draw.polygon(xy, outline=1, fill=1)

    return np.array(mask_img, dtype=np.uint8)


def extract_bbox_crop(image: Image.Image, bbox_xyxy: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox_xyxy
    assert x1 <= x2 and y1 <= y2, f"Bad bbox passed to crop: {bbox_xyxy}"
    return image.crop(bbox_xyxy)


def extract_masked_crop(
    image: Image.Image,
    mask: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    background_value: int = 0
) -> Image.Image:
    """
    Extract a crop using the given bounding box, but remove all pixels outside
    the leaf mask.

    The returned crop keeps the same rectangle as the bbox crop, but pixels
    outside the selected leaf are replaced with the background value.

    Args:
        image: PIL image
        mask: full-image binary mask (H, W)
        bbox_xyxy: tuple (x1, y1, x2, y2)
        background_value: grayscale/RGB fill value, default black

    Returns:
        PIL image of the masked crop
    """
    x1, y1, x2, y2 = bbox_xyxy

    # Convert image to numpy for masking
    image_np = np.array(image)

    # Crop image and mask to the bbox region
    crop_np = image_np[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Ensure crop_mask is boolean for indexing
    crop_mask_bool = crop_mask.astype(bool)

    # Handle grayscale or RGB
    crop_np[~crop_mask_bool] = background_value # handle both branches in future if necessary

    return Image.fromarray(crop_np)


def load_tracker_csv(tracker_csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load tracker.csv into a lookup dictionary keyed by filename.

    This function is intentionally generic because tracker.csv may evolve.
    All columns are preserved as strings.

    Returns:
        dict mapping filename -> row dict
    """
    tracker_lookup: Dict[str, Dict[str, str]] = {}

    if not tracker_csv_path.exists():
        return tracker_lookup

    with tracker_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename")
            if filename:
                tracker_lookup[filename] = row

    return tracker_lookup


# -----------------------------------------------------------------------------
# COCO loading and indexing
# -----------------------------------------------------------------------------

def load_coco_split(split_dir: Path, split_name: str) -> Tuple[List[ImageRecord], List[AnnotationRecord], Dict[int, str]]:
    """
    Load a single COCO split (train or val).

    Args:
        split_dir: path like ./datasets/agai_correct/v3/coco/train
        split_name: "train" or "val"

    Returns:
        images: list of ImageRecord
        annotations: list of AnnotationRecord
        categories: dict mapping category_id -> category_name
    """
    json_path = split_dir / "_annotations.coco.json"
    coco = load_json(json_path)

    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    images: List[ImageRecord] = []
    for img in coco.get("images", []):
        file_name = img["file_name"]
        image_path = split_dir / file_name

        images.append(
            ImageRecord(
                image_id=img["id"],
                file_name=file_name,
                width=img["width"],
                height=img["height"],
                extra_name=img.get("extra", {}).get("name"),
                split=split_name,
                image_path=image_path
            )
        )

    annotations: List[AnnotationRecord] = []
    for ann in coco.get("annotations", []):
        annotations.append(
            AnnotationRecord(
                annotation_id=ann["id"],
                image_id=ann["image_id"],
                category_id=ann["category_id"],
                bbox=ann["bbox"],
                area=ann["area"],
                segmentation=ann["segmentation"],
                iscrowd=ann.get("iscrowd", 0)
            )
        )

    return images, annotations, categories


def build_annotation_index(annotations: List[AnnotationRecord]) -> Dict[int, List[AnnotationRecord]]:
    """
    Build a lookup from image_id -> list of annotations.
    """
    ann_index: Dict[int, List[AnnotationRecord]] = defaultdict(list)
    for ann in annotations:
        ann_index[ann.image_id].append(ann)
    return ann_index


def find_leaf_category_id(categories: Dict[int, str]) -> int:
    """
    Find the category ID corresponding to 'leaf'.

    Raises:
        ValueError if the class is not found.
    """
    for category_id, name in categories.items():
        if name.lower() == "leaf":
            return category_id
    raise ValueError("Could not find a 'leaf' category in COCO categories.")


# -----------------------------------------------------------------------------
# Core logic for largest leaf selection
# -----------------------------------------------------------------------------

def select_largest_leaf(
    image_record: ImageRecord,
    image_annotations: List[AnnotationRecord],
    leaf_category_id: int,
    actual_width: Optional[int] = None,
    actual_height: Optional[int] = None,
) -> Optional[LargestLeafResult]:
    """
    Select the largest leaf annotation for a single image.

    The largest leaf is chosen by the COCO 'area' field.
    actual_width/actual_height override image_record dimensions to handle
    COCO JSON metadata that may have swapped or incorrect dimensions.

    Returns:
        LargestLeafResult or None if no valid leaf annotation exists
    """
    leaf_annotations = [
        ann for ann in image_annotations
        if ann.category_id == leaf_category_id and ann.iscrowd == 0
    ]

    if not leaf_annotations:
        return None

    largest_ann = max(leaf_annotations, key=lambda ann: ann.area)

    w = actual_width  if actual_width  is not None else image_record.width
    h = actual_height if actual_height is not None else image_record.height

    try:
        mask = segmentation_to_mask(
            segmentation=largest_ann.segmentation,
            image_width=w,
            image_height=h
        )
    except Exception as e:
        print(f"[WARNING] Failed to create mask for {image_record.file_name}: {e}")
        return None

    bbox_xyxy = coco_bbox_to_xyxy(
        bbox=largest_ann.bbox,
        image_width=w,
        image_height=h
    )

    return LargestLeafResult(
        annotation=largest_ann,
        mask=mask,
        bbox_xyxy=bbox_xyxy
    )

# -----------------------------------------------------------------------------
# Main processing routine
# -----------------------------------------------------------------------------

def process_dataset(
    dataset_root: Path,
    tracker_csv_path: Path,
    output_root: Path,
    background_value: int = 0
) -> None:
    """
    Full pipeline:
        - load train and val COCO annotations
        - select largest leaf per image
        - save bbox and masked crops
        - write metadata CSV
    """
    coco_root = dataset_root / "coco"
    train_dir = coco_root / "train"
    val_dir = coco_root / "val"

    # Output directories
    bbox_train_dir = output_root / "bbox" / "train"
    bbox_val_dir = output_root / "bbox" / "val"
    masked_train_dir = output_root / "masked" / "train"
    masked_val_dir = output_root / "masked" / "val"

    ensure_dir(bbox_train_dir)
    ensure_dir(bbox_val_dir)
    ensure_dir(masked_train_dir)
    ensure_dir(masked_val_dir)

    # Load tracker.csv if present
    tracker_lookup = load_tracker_csv(tracker_csv_path)

    # Load train + val COCO
    train_images, train_annotations, train_categories = load_coco_split(train_dir, "train")
    val_images, val_annotations, val_categories = load_coco_split(val_dir, "val")

    # Sanity-check categories across splits
    if train_categories != val_categories:
        raise ValueError("Train and val category mappings do not match.")

    categories = train_categories
    leaf_category_id = find_leaf_category_id(categories)

    # Build annotation indices
    train_ann_index = build_annotation_index(train_annotations)
    val_ann_index = build_annotation_index(val_annotations)

    # Combine for unified processing
    all_images = train_images + val_images
    all_ann_index = {}
    all_ann_index.update(train_ann_index)
    for k, v in val_ann_index.items():
        all_ann_index[k] = v

    metadata_rows: List[Dict[str, Any]] = []

    for image_record in all_images:
        if not image_record.image_path.exists():
            print(f"[WARNING] Missing image file: {image_record.image_path}")
            continue

        ppm_bin, sample_id, base_stub = parse_filename_metadata(image_record.file_name)
        if ppm_bin is None:
            print(f"[WARNING] Could not parse filename metadata: {image_record.file_name}")
            continue

        # Open source image FIRST so we have actual dimensions
        with Image.open(image_record.image_path) as img:
            image = img.convert("RGB")
        actual_w, actual_h = image.size

        image_annotations = all_ann_index.get(image_record.image_id, [])
        largest_leaf = select_largest_leaf(
            image_record=image_record,
            image_annotations=image_annotations,
            leaf_category_id=leaf_category_id,
            actual_width=actual_w,
            actual_height=actual_h,
        )

        if largest_leaf is None:
            print(f"[WARNING] No valid leaf annotation found for {image_record.file_name}")
            continue
        
        # Create crop variants
        bbox_crop = extract_bbox_crop(image, largest_leaf.bbox_xyxy)
        masked_crop = extract_masked_crop(
            image=image,
            mask=largest_leaf.mask,
            masked_crop = extract_masked_crop(
                image=image,
                mask=largest_leaf.mask,
                bbox_xyxy=largest_leaf.bbox_xyxy,
                background_value=background_value
            )
        )
        
        # Build output filenames
        # Keep naming explicit and traceable to the source image.
        stem = Path(image_record.file_name).stem
        bbox_filename = f"{stem}_largest_leaf_bbox.png"
        masked_filename = f"{stem}_largest_leaf_masked.png"

        if image_record.split == "train":
            bbox_out_path = bbox_train_dir / bbox_filename
            masked_out_path = masked_train_dir / masked_filename
        else:
            bbox_out_path = bbox_val_dir / bbox_filename
            masked_out_path = masked_val_dir / masked_filename

        # Save crops
        bbox_crop.save(bbox_out_path)
        masked_crop.save(masked_out_path)

        # Optional tracker join
        tracker_row = tracker_lookup.get(image_record.file_name, {})

        # Metadata row for downstream training
        metadata_rows.append({
            "source_filename": image_record.file_name,
            "source_extra_name": image_record.extra_name or "",
            "split": image_record.split,
            "ppm_bin": ppm_bin,
            "sample_id": sample_id if sample_id is not None else "",
            "width": image_record.width,
            "height": image_record.height,
            "largest_leaf_annotation_id": largest_leaf.annotation.annotation_id,
            "largest_leaf_area": largest_leaf.annotation.area,
            "largest_leaf_bbox_x1": largest_leaf.bbox_xyxy[0],
            "largest_leaf_bbox_y1": largest_leaf.bbox_xyxy[1],
            "largest_leaf_bbox_x2": largest_leaf.bbox_xyxy[2],
            "largest_leaf_bbox_y2": largest_leaf.bbox_xyxy[3],
            "bbox_crop_path": str(bbox_out_path.resolve()),
            "masked_crop_path": str(masked_out_path.resolve()),
            **tracker_row,  # preserve any extra columns from tracker.csv
        })

    # Save metadata CSV
    metadata_csv_path = output_root / "leaf_crop_metadata.csv"
    write_metadata_csv(metadata_csv_path, metadata_rows)

    print("\nDone.")
    print(f"Saved metadata CSV to: {metadata_csv_path}")
    print(f"Saved bbox crops to:      {output_root / 'bbox'}")
    print(f"Saved masked crops to:    {output_root / 'masked'}")
    print(f"Total processed images:   {len(metadata_rows)}")


def write_metadata_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write metadata rows to CSV.

    Dynamically unions all keys across rows to avoid losing any tracker.csv fields.
    """
    ensure_dir(csv_path.parent)

    if not rows:
        print(f"[WARNING] No metadata rows to write for {csv_path}")
        return

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    # Put common fields first for readability
    preferred_order = [
        "source_filename",
        "source_extra_name",
        "split",
        "ppm_bin",
        "sample_id",
        "width",
        "height",
        "largest_leaf_annotation_id",
        "largest_leaf_area",
        "largest_leaf_bbox_x1",
        "largest_leaf_bbox_y1",
        "largest_leaf_bbox_x2",
        "largest_leaf_bbox_y2",
        "bbox_crop_path",
        "masked_crop_path",
    ]

    remaining_keys = [k for k in sorted(all_keys) if k not in preferred_order]
    fieldnames = preferred_order + remaining_keys

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract largest-leaf bbox and masked crops from COCO annotations."
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./datasets/agai_correct/v3",
        help="Path to dataset root containing coco/train, coco/val, and tracker.csv"
    )

    parser.add_argument(
        "--tracker-csv",
        type=str,
        default="./datasets/agai_correct/v3/tracker.csv",
        help="Path to tracker.csv (optional; missing file is handled)"
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="./datasets/processed_leaf_crops",
        help="Directory where processed crops and metadata CSV will be saved"
    )

    parser.add_argument(
        "--background-value",
        type=int,
        default=0,
        help="Background fill value for masked crops (default=0, black)"
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point.
    """
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    tracker_csv_path = Path(args.tracker_csv)
    output_root = Path(args.output_root)

    process_dataset(
        dataset_root=dataset_root,
        tracker_csv_path=tracker_csv_path,
        output_root=output_root,
        background_value=args.background_value
    )


if __name__ == "__main__":
    main()