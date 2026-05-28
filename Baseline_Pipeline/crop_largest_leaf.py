import argparse
import json
from pathlib import Path

import cv2


JSON_PATH = Path(
    "/mnt/c/Users/david/OneDrive - University of Central Florida/"
    "AgAI_REU_2026/Datasets/AgAI Correct/V3 - 2026-03-25 1415/"
    "COCO JSON Masks/train/_annotations.coco.json"
)

IMAGE_DIR = JSON_PATH.parent
OUTPUT_DIR = IMAGE_DIR / "largest_leaf_crop"

LEAF_CATEGORY_ID = 1


def load_coco_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image_info(images, target_file_name):
    for image_info in images:
        if image_info["file_name"] == target_file_name:
            return image_info

    return None


def get_leaf_annotations(annotations, image_id):
    return [
        ann for ann in annotations
        if ann["image_id"] == image_id and ann["category_id"] == LEAF_CATEGORY_ID
    ]


def select_leaf_by_area_rank(leaf_annotations, area_rank):
    leaf_annotations_sorted = sorted(
        leaf_annotations,
        key=lambda ann: ann["area"],
        reverse=True
    )

    if area_rank > len(leaf_annotations_sorted):
        return None

    return leaf_annotations_sorted[area_rank - 1]


def bbox_to_int_coords(bbox):
    x, y, w, h = bbox

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    return x1, y1, x2, y2


def crop_leaf(image, bbox):
    x1, y1, x2, y2 = bbox_to_int_coords(bbox)
    return image[y1:y2, x1:x2]


def save_crop(image, bbox, output_path):
    cropped_leaf = crop_leaf(image, bbox)
    cv2.imwrite(str(output_path), cropped_leaf)


def process_image_by_rank(image_info, annotations, area_rank, output_dir, output_prefix=""):
    image_id = image_info["id"]
    file_name = image_info["file_name"]

    image_path = IMAGE_DIR / file_name
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not load image: {image_path}")
        return

    leaf_annotations = get_leaf_annotations(annotations, image_id)

    if not leaf_annotations:
        print(f"No leaf annotations found for {file_name}")
        return

    selected_leaf = select_leaf_by_area_rank(leaf_annotations, area_rank)

    if selected_leaf is None:
        print(
            f"{file_name} only has {len(leaf_annotations)} leaf annotations. "
            f"Cannot select rank {area_rank}."
        )
        return

    output_name = f"{output_prefix}{file_name}"
    output_path = output_dir / output_name

    save_crop(image, selected_leaf["bbox"], output_path)

    print(f"Saved: {output_path}")
    print(f"Selected rank: {area_rank}")
    print(f"Selected area: {selected_leaf['area']}")
    print(f"Selected bbox: {selected_leaf['bbox']}")


def process_image_by_manual_bbox(image_info, manual_bbox, output_dir):
    file_name = image_info["file_name"]

    image_path = IMAGE_DIR / file_name
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not load image: {image_path}")
        return

    output_name = f"manual_{file_name}"
    output_path = output_dir / output_name

    save_crop(image, manual_bbox, output_path)

    print(f"Saved: {output_path}")
    print(f"Manual bbox: {manual_bbox}")


def process_all_images(coco, area_rank, output_dir):
    images = coco["images"]
    annotations = coco["annotations"]

    for image_info in images:
        process_image_by_rank(
            image_info=image_info,
            annotations=annotations,
            area_rank=area_rank,
            output_dir=output_dir
        )


def process_target_image(coco, target_file_name, area_rank, manual_bbox, output_dir):
    images = coco["images"]
    annotations = coco["annotations"]

    target_image_info = find_image_info(images, target_file_name)

    if target_image_info is None:
        print(f"Target file not found in COCO JSON: {target_file_name}")
        return

    if manual_bbox is not None:
        process_image_by_manual_bbox(
            image_info=target_image_info,
            manual_bbox=manual_bbox,
            output_dir=output_dir
        )
    else:
        process_image_by_rank(
            image_info=target_image_info,
            annotations=annotations,
            area_rank=area_rank,
            output_dir=output_dir,
            output_prefix=f"rank_{area_rank}_"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop leaf images using COCO leaf annotations or manual bbox."
    )

    parser.add_argument(
        "--mode",
        choices=["all", "target"],
        required=True,
        help="Use 'all' to process every image or 'target' to process one image."
    )

    parser.add_argument(
        "--target-file",
        type=str,
        default=None,
        help="File name to process when using target mode."
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Leaf area rank to select. 1 = biggest, 2 = second biggest, etc."
    )

    parser.add_argument(
        "--manual",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Manual bbox crop as x y w h. Only used in target mode."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    coco = load_coco_json(JSON_PATH)

    if args.mode == "all":
        area_rank = args.rank if args.rank is not None else 1

        process_all_images(
            coco=coco,
            area_rank=area_rank,
            output_dir=OUTPUT_DIR
        )

    elif args.mode == "target":
        if args.target_file is None:
            print("Error: --target-file is required when using --mode target")
            return

        area_rank = args.rank if args.rank is not None else 2

        process_target_image(
            coco=coco,
            target_file_name=args.target_file,
            area_rank=area_rank,
            manual_bbox=args.manual,
            output_dir=OUTPUT_DIR
        )


if __name__ == "__main__":
    main()