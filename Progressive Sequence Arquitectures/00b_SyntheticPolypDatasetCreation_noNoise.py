#python3 00_SyntheticPolypDatasetCreation.py --image_size 64  --train_count 512 --val_count 126 --test_count 126 --output_dir ./Datasets/dataset_64by64_Train512Val126Test126

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
parser = argparse.ArgumentParser(description="Synthetic Polyp Dataset Generator (Visible Ellipses)")

parser.add_argument("--image_size", type=int, default=8,
                    help="Image width/height in pixels (default: 8)")
parser.add_argument("--train_count", type=int, default=700,
                    help="Number of images for training (default: 700)")
parser.add_argument("--val_count", type=int, default=150,
                    help="Number of images for validation (default: 150)")
parser.add_argument("--test_count", type=int, default=150,
                    help="Number of images for testing (default: 150)")
parser.add_argument("--output_dir", type=str,
                    default="./dataset",
                    help="Output dataset directory")

args = parser.parse_args()

IMAGE_SIZE = args.image_size
OUTPUT_DIR = args.output_dir

# Map the counts to a dictionary for easy iteration
SPLIT_SIZES = {
    "train": args.train_count,
    "val": args.val_count,
    "test": args.test_count
}

# =========================
# Fixed parameters
# =========================
GRAYSCALE_LEVELS = 8
SPLITS = ["train", "val", "test"]
CLASSES = ["with_ellipse", "without_ellipse"]

# =========================
# Create directories
# =========================
for split in SPLITS:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# =========================
# Image generation functions
# =========================
def create_smooth_gradient_background(size):
    corners = np.random.randint(0, GRAYSCALE_LEVELS, size=4) * (255 // (GRAYSCALE_LEVELS - 1))
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    top = corners[0] * (1 - x) + corners[1] * x
    bottom = corners[2] * (1 - x) + corners[3] * x
    gradient = top * (1 - y) + bottom * y
    return np.round(gradient).astype(np.float32)

def add_ellipse_unique_color(img_array, size):
    """
    Adds an ellipse by picking a center point inside the image.
    This ensures the ellipse is clearly visible and not 'lost' off-screen.
    """
    temp_img = Image.fromarray(img_array.astype(np.uint8), 'L')
    draw = ImageDraw.Draw(temp_img)

    # 1. Determine size (minimum 2 pixels, maximum half the image)
    e_width = random.randint(2, max(3, size // 2))
    e_height = random.randint(2, max(3, size // 2))

    # 2. Pick a center point that is definitely inside the image
    # This guarantees at least 50% visibility
    center_x = random.randint(0, size - 1)
    center_y = random.randint(0, size - 1)

    # 3. Calculate the bounding box based on that center
    x0 = center_x - e_width // 2
    y0 = center_y - e_height // 2
    x1 = x0 + e_width
    y1 = y0 + e_height

    # 4. Color logic
    background_colors = set(np.unique(img_array.astype(np.uint8)))
    possible_colors = set(i * (255 // (GRAYSCALE_LEVELS - 1)) for i in range(GRAYSCALE_LEVELS))
    available_colors = list(possible_colors - background_colors)

    fill_value = int(random.choice(available_colors) if available_colors else max(background_colors))

    draw.ellipse([x0, y0, x1, y1], fill=fill_value)
    return np.array(temp_img).astype(np.float32)

def apply_gaussian_noise(img_array, mean=0, std=0):
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255)
    return noisy_img.astype(np.uint8)

def create_image(has_ellipse):
    img_array = create_smooth_gradient_background(IMAGE_SIZE)
    if has_ellipse:
        img_array = add_ellipse_unique_color(img_array, IMAGE_SIZE)
    img_array = apply_gaussian_noise(img_array)
    return img_array

# =========================
# Generate dataset
# =========================
datasets = {"train": [], "val": [], "test": []}
image_id = 0

for split in SPLITS:
    total_for_split = SPLIT_SIZES[split]
    
    # Logic to handle odd numbers
    num_with = total_for_split // 2
    num_without = total_for_split - num_with 

    class_tasks = [(True, num_with), (False, num_without)]

    for has_ellipse, count in class_tasks:
        class_name = "with_ellipse" if has_ellipse else "without_ellipse"

        for _ in tqdm(range(count), desc=f"{split} - {class_name}"):
            img_array = create_image(has_ellipse)
            img = Image.fromarray(img_array, 'L')

            img_name = f"{image_id}.png"
            img_path = os.path.join(OUTPUT_DIR, split, class_name, img_name)
            img.save(img_path)

            datasets[split].append({
                "filename": img_path,
                "label": int(has_ellipse)
            })
            image_id += 1

# =========================
# Save labels
# =========================
for split in SPLITS:
    if datasets[split]:
        df = pd.DataFrame(datasets[split])
        df.to_csv(os.path.join(OUTPUT_DIR, f"labels_{split}.csv"), index=False)

print(f"\nDataset generation complete!")
print(f"Summary: Train={args.train_count}, Val={args.val_count}, Test={args.test_count}")
