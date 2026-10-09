"""Create an image manifest for the Kocide treatment folders (run in WSL)."""

import csv
from pathlib import Path
from pillow_heif import register_heif_opener

register_heif_opener()

DATASET_DIR = Path(
    "/home/davidorjuela/dev/agai-reu-pesticide/datasets/kocide_application_2025"
)
OUTPUT_CSV = DATASET_DIR / "nominal_treatment.csv"
TREATMENTS = (400, 600, 800, 1000)
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".heic", ".heif",
}


def main():
    # Validate all four folders before writing the CSV.
    for treatment in TREATMENTS:
        folder = DATASET_DIR / f"{treatment} ppm"
        if not folder.is_dir():
            raise FileNotFoundError(f"Treatment folder not found: {folder}")

    total = 0
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "nominal_treatment"])

        for treatment in TREATMENTS:
            folder = DATASET_DIR / f"{treatment} ppm"
            count = 0
            for image in sorted(folder.rglob("*")):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    writer.writerow([str(image.absolute()), treatment])
                    count += 1
            total += count
            print(f"{treatment} ppm: {count} images")

    print(f"Saved {total} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
