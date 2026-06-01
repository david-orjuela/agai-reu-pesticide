'''
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen

This file will preprocess the V3 - 2026 AgAI Correct dataset images into a ground-truth, training-ready file.
'''
import csv
from pathlib import Path
import pandas as pd

header = ["image_path","sample_id","nominal_treatment","total_mg","area_cm2","mg_cm2","residue_bin"]
new_excel_data = [
    header,
]

# Parse image filename (WSL compatible)
images_folder =   Path("/home/david/dev/agai-reu-2026/datasets/agai_correct/v3/coco_json_masks")
if images_folder.exists():
    print("Images path exists. Continuing...")

icp_path = Path("/home/david/dev/agai-reu-2026/datasets/agai_correct/v3/icp_data.csv")
if icp_path.exists():
    print("ICP path exists. Continuing...")

print("Reading and cleaning ICP CSV file...")
df = pd.read_csv(icp_path, encoding="cp1252")

# Clean column names just in case there are hidden spaces
df.columns = df.columns.str.strip()

# Compute tertiles once, not inside every loop
print("Computing tertiles...")
t1 = df["mg / cm²"].quantile(1/3)
t2 = df["mg / cm²"].quantile(2/3)
print(f"T1: {t1:.4f} \t T2: {t2:.4f}")

def calc_bin_tertiles(mg_cm2):
    if mg_cm2 < t1:
        return "low"
    elif mg_cm2 < t2:
        return "medium"
    else:
        return "high"
    
print("Looping through images in dataset...")
for img in images_folder.rglob("*.jpg"):
    if "largest_leaf_crop" in img.parts or "largest_leaf_bbox_debug" in img.parts:
        continue

    stem = img.stem  # ICP_800ppm_1_jpg...
    parts = stem.split("_")

    nominal_treatment = parts[1].replace("ppm", "")  # 800
    if nominal_treatment == "1K":
        nominal_treatment = 1000

    sample_num = parts[2]                            # 1
    sample_id = f"{nominal_treatment} ppm {sample_num}" # ICP_800ppm_1_jpg becomes 800 ppm 1, max number is 9

    match = df.loc[df["Name"] == sample_id]
    
    if match.empty:
        print(f"No ICP match found for {stem} -> {sample_id}")
        continue
    
    total_mg = match["Total mg"].iloc[0]
    area_cm2 = match["Area cm²"].iloc[0]
    mg_cm2 = match["mg / cm²"].iloc[0]
    residue_bin = calc_bin_tertiles(mg_cm2) # puts ppm into 3 bins, tertiles
    
    cropped_img = img.parent / "largest_leaf_crop" / img.name

    if not cropped_img.exists():
        print(f"Cropped image missing for {img.parent.name}/{img.name}")
        continue

    new_image_path = str(cropped_img)

    new_row = [
        new_image_path,
        sample_id,
        nominal_treatment,
        total_mg,
        area_cm2,
        mg_cm2,
        residue_bin
    ]

    new_excel_data.append(new_row)

master_path = Path("/home/david/dev/agai-reu-2026/datasets/agai_correct/v3/master_icp.csv").resolve()
with open(master_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(new_excel_data)

print(f"New Master ICP CSV created. File at: {master_path.as_uri()}")