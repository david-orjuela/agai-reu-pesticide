import json
from pathlib import Path

import cv2


JSON_PATH = Path(
    "/mnt/c/Users/david/OneDrive - University of Central Florida/"
    "AgAI_REU_2026/Datasets/AgAI Correct/V3 - 2026-03-25 1415/"
    "COCO JSON Masks/train/_annotations.coco.json"
)

IMAGE_DIR = JSON_PATH.parent
OUTPUT_DIR = IMAGE_DIR / "largest_leaf_bbox_debug"

LEAF_CATEGORY_ID = 1

# Hardcoded debug settings
TARGET_FILE_NAME = "ICP_600ppm_3_jpg.rf.f824ab9238a2b11e44f40fecdfa4190d.jpg"
TARGET_AREA_RANK = 2  # 1 = biggest, 2 = second biggest, 3 = third biggest, etc.


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    for image_info in images:
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        # ---------------------------------------------------------------------
        # ORIGINAL FULL-DATASET MODE:
        # Uncomment this block if you want to process every image again.
        # ---------------------------------------------------------------------
        # image_path = IMAGE_DIR / file_name
        # image = cv2.imread(str(image_path))
        #
        # if image is None:
        #     print(f"Could not load image: {image_path}")
        #     continue
        #
        # leaf_annotations = [
        #     ann for ann in annotations
        #     if ann["image_id"] == image_id and ann["category_id"] == LEAF_CATEGORY_ID
        # ]
        #
        # if not leaf_annotations:
        #     print(f"No leaf annotations found for {file_name}")
        #     continue
        #
        # largest_leaf = max(leaf_annotations, key=lambda ann: ann["area"])
        #
        # x, y, w, h = largest_leaf["bbox"]
        #
        # x1 = int(round(x))
        # y1 = int(round(y))
        # x2 = int(round(x + w))
        # y2 = int(round(y + h))
        #
        # cv2.rectangle(
        #     image,
        #     (x1, y1),
        #     (x2, y2),
        #     color=(0, 0, 255),
        #     thickness=8
        # )
        #
        # output_path = OUTPUT_DIR / file_name
        # cv2.imwrite(str(output_path), image)
        #
        # print(f"Saved: {output_path}")

        # ---------------------------------------------------------------------
        # HARDCODED SINGLE-IMAGE MODE:
        # This only processes TARGET_FILE_NAME and selects the nth biggest leaf.
        # ---------------------------------------------------------------------
        if file_name != TARGET_FILE_NAME:
            continue

        image_path = IMAGE_DIR / file_name
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        leaf_annotations = [
            ann for ann in annotations
            if ann["image_id"] == image_id and ann["category_id"] == LEAF_CATEGORY_ID
        ]

        if not leaf_annotations:
            print(f"No leaf annotations found for {file_name}")
            continue

        leaf_annotations_sorted = sorted(
            leaf_annotations,
            key=lambda ann: ann["area"],
            reverse=True
        )

        if TARGET_AREA_RANK > len(leaf_annotations_sorted):
            print(
                f"{file_name} only has {len(leaf_annotations_sorted)} leaf annotations. "
                f"Cannot select rank {TARGET_AREA_RANK}."
            )
            continue

        selected_leaf = leaf_annotations_sorted[TARGET_AREA_RANK - 1]

        x, y, w, h = selected_leaf["bbox"]

        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color=(0, 0, 255),
            thickness=8
        )

        output_path = OUTPUT_DIR / f"rank_{TARGET_AREA_RANK}_{file_name}"
        cv2.imwrite(str(output_path), image)

        print(f"Saved: {output_path}")
        print(f"Selected annotation area: {selected_leaf['area']}")
        print(f"Selected bbox: {selected_leaf['bbox']}")

        break


if __name__ == "__main__":
    main()