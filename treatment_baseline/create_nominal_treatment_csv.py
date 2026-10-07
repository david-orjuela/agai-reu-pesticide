"""Create an image manifest for the Kocide treatment folders (run in WSL)."""

import csv
from pathlib import Path


DATASET_DIR = Path(
    "/mnt/c/Users/david/OneDrive - University of Central Florida/"
    "agai-reu-pesticide/datasets/!Kocide Application 2025"
)
OUTPUT_CSV = "/home/davidorjuela/dev/agai-reu-pesticide/treatment_baseline"
TREATMENTS = (400, 600, 800, 1000)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
