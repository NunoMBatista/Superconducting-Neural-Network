#python3 00a_SyntheticPolypDatasetCreation_SimpleBW.py --image_size 64  --train_count 512 --val_count 126 --test_count 126 --output_dir ./Datasets/dataset_64by64_Train512Val126Test126

import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

# =========================
# Argument parser
# =========================
parser = argparse.ArgumentParser(description="Simple Black & White Polyp Dataset Generator")

parser.add_argument("--image_size", type=int, default=64, help="Image size (default: 64)")
parser.add_argument("--train_count", type=int, default=512, help="Train count")
parser.add_argument("--val_count", type=int, default=126, help="Val count")
parser.add_argument("--test_count", type=int, default=126, help="Test count")
parser.add_argument("--output_dir", type=str, default="./Datasets/simple_bw_dataset", help="Output directory")

args = parser.parse_args()

IMAGE_SIZE = args.image_size
OUTPUT_DIR = args.output_dir
SPLIT_SIZES = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
SPLITS = ["train", "val", "test"]
CLASSES = ["with_ellipse", "without_ellipse"]

# Create directories
for split in SPLITS:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# =========================
# Simplified Generation Functions
# =========================

def create_simple_image(has_ellipse):
    # 1. Create solid white background (255)
    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color=255)
    
    if has_ellipse:
        draw = ImageDraw.Draw(img)
        
        # 2. Determine size (minimum 4 pixels to be clearly visible)
        e_width = random.randint(4, max(5, IMAGE_SIZE // 3))
        e_height = random.randint(4, max(5, IMAGE_SIZE // 3))

        # 3. Pick a center point
        center_x = random.randint(0, IMAGE_SIZE - 1)
        center_y = random.randint(0, IMAGE_SIZE - 1)

        # 4. Bounding box for the black ellipse (0)
        x0, y0 = center_x - e_width // 2, center_y - e_height // 2
        x1, y1 = x0 + e_width, y0 + e_height
        
        draw.ellipse([x0, y0, x1, y1], fill=0)
        
    return img

# =========================
# Generate dataset
# =========================
datasets = {"train": [], "val": [], "test": []}
image_id = 0

for split in SPLITS:
    total_for_split = SPLIT_SIZES[split]
    num_with = total_for_split // 2
    num_without = total_for_split - num_with 

    for has_ellipse, count in [(True, num_with), (False, num_without)]:
        class_name = "with_ellipse" if has_ellipse else "without_ellipse"

        for _ in tqdm(range(count), desc=f"{split} - {class_name}"):
            img = create_simple_image(has_ellipse)
            
            img_name = f"{image_id}.png"
            img_path = os.path.join(OUTPUT_DIR, split, class_name, img_name)
            img.save(img_path)

            datasets[split].append({
                "filename": img_path,
                "label": int(has_ellipse)
            })
            image_id += 1

# Save labels
for split in SPLITS:
    if datasets[split]:
        pd.DataFrame(datasets[split]).to_csv(os.path.join(OUTPUT_DIR, f"labels_{split}.csv"), index=False)

print(f"\nSimple BW Dataset generation complete in: {OUTPUT_DIR}")
