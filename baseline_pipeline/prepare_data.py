"""
Create a portrait-oriented copy of the largest-leaf-crop dataset.

For every row in master_icp.csv:
1. Find the corresponding image in largest_leaf_crop by filename.
2. Apply EXIF orientation.
3. Rotate 90 degrees clockwise if width > height.
4. Verify the saved image has height > width.
5. Save the image under ./anthony/largest_leaf_crop/.
6. Update image_path in a copied master_icp.csv.

The original images and original CSV are never modified.
"""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/davidorjuela/dev/agai-reu-pesticide")

SOURCE_IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "agai_correct"
    / "batch_2_v1"
    / "coco"
    / "train"
    / "largest_leaf_crop"
)

SOURCE_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "agai_correct"
    / "batch_2_v1"
    / "master_icp.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "anthony"
OUTPUT_IMAGE_DIR = OUTPUT_ROOT / "largest_leaf_crop"
OUTPUT_CSV = OUTPUT_ROOT / "master_icp.csv"

IMAGE_PATH_COLUMN = "image_path"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def save_image(image, output_path):
    """Save an image while keeping the original file format."""
    suffix = output_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        image.save(
            output_path,
            quality=100,
            subsampling=0,
        )
    else:
        image.save(output_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """Copy the CSV-linked leaf crops into portrait orientation."""
    if not SOURCE_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Source image directory does not exist:\n{SOURCE_IMAGE_DIR}"
        )

    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(
            f"Source CSV does not exist:\n{SOURCE_CSV}"
        )

    metadata = pd.read_csv(SOURCE_CSV)

    if IMAGE_PATH_COLUMN not in metadata.columns:
        raise ValueError(
            f"CSV is missing required column: {IMAGE_PATH_COLUMN}"
        )

    if metadata[IMAGE_PATH_COLUMN].isna().any():
        raise ValueError(
            f"CSV contains missing values in {IMAGE_PATH_COLUMN}"
        )

    OUTPUT_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rotated = 0
    already_portrait = 0
    square = 0
    output_paths = []

    print(f"CSV rows: {len(metadata)}")
    print(f"Source images: {SOURCE_IMAGE_DIR}")
    print(f"Output images: {OUTPUT_IMAGE_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print()

    for csv_image_path in tqdm(
        metadata[IMAGE_PATH_COLUMN],
        desc="Orienting leaves",
    ):
        filename = Path(str(csv_image_path)).name
        source_path = SOURCE_IMAGE_DIR / filename
        output_path = OUTPUT_IMAGE_DIR / filename

        if not source_path.is_file():
            raise FileNotFoundError(
                "Image referenced by the CSV was not found:\n"
                f"CSV path: {csv_image_path}\n"
                f"Expected local path: {source_path}"
            )

        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened)

            width, height = image.size

            if width > height:
                image = image.rotate(
                    -90,
                    expand=True,
                )
                rotated += 1

            elif height > width:
                already_portrait += 1

            else:
                square += 1
                raise ValueError(
                    "A square image cannot satisfy height > width by rotation:\n"
                    f"{source_path}\n"
                    f"size={width}x{height}"
                )

            final_width, final_height = image.size

            if final_height <= final_width:
                raise ValueError(
                    "Orientation verification failed:\n"
                    f"{source_path}\n"
                    f"final size={final_width}x{final_height}"
                )

            save_image(
                image,
                output_path,
            )

        # Re-open the actual saved file to verify what is on disk.
        with Image.open(output_path) as saved:
            saved_width, saved_height = saved.size

        if saved_height <= saved_width:
            raise ValueError(
                "Saved image failed portrait verification:\n"
                f"{output_path}\n"
                f"saved size={saved_width}x{saved_height}"
            )

        output_paths.append(
            str(output_path.resolve())
        )

    if len(output_paths) != len(metadata):
        raise RuntimeError(
            "Output-path count does not match CSV row count."
        )

    if len(set(output_paths)) != len(output_paths):
        raise ValueError(
            "Multiple CSV rows resolve to the same output image path."
        )

    metadata[IMAGE_PATH_COLUMN] = output_paths

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # Final CSV verification.
    saved_metadata = pd.read_csv(OUTPUT_CSV)

    if len(saved_metadata) != len(metadata):
        raise RuntimeError(
            "Saved CSV row count changed unexpectedly."
        )

    missing_outputs = [
        path
        for path in saved_metadata[IMAGE_PATH_COLUMN]
        if not Path(path).is_file()
    ]

    if missing_outputs:
        raise FileNotFoundError(
            f"{len(missing_outputs)} output images referenced by the "
            "new CSV are missing."
        )

    print()
    print("Complete")
    print(f"Rows processed: {len(metadata)}")
    print(f"Rotated: {rotated}")
    print(f"Already portrait: {already_portrait}")
    print(f"Square: {square}")
    print(f"Images: {OUTPUT_IMAGE_DIR}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()