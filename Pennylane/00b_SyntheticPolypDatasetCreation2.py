#python generate_dataset.py --image_size 8  --num_images 2048  --output_dir /home/amorgado/Pennylane_codes/Datasets/dataset_2048_8by8

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
def parse_args():
    parser = argparse.ArgumentParser(description="Original Pink Polyp Dataset Generator for QNN")
    parser.add_argument("--image_size", type=int, default=128, help="Image size (default: 128)")
    parser.add_argument("--num_images", type=int, default=1000, help="Total images (default: 1000)")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output directory")
    return parser.parse_args()

# =================================================================
# Part 1: Original Enhanced Synthetic Colon Image Generation Logic
# =================================================================

def generate_colon_background(img_size=256, noise_factor=0.1):
    base_color = np.array([210, 140, 140])
    img = np.ones((img_size, img_size, 3))
    y, x = np.ogrid[:img_size, :img_size]
    center = img_size // 2
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    dist_from_center = dist_from_center / (np.sqrt(2) * center)
    for c in range(3):
        img[:, :, c] = base_color[c] / 255.0 * (0.6 + 0.4 * dist_from_center)
    noise = np.random.randn(img_size, img_size, 3) * noise_factor
    img += noise
    return np.clip(img, 0, 1)

def add_colon_folds(img, num_folds=6, fold_width_range=(8, 15)):
    img_size = img.shape[0]
    center = img_size // 2
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    for i in range(1, num_folds + 1):
        radius = (img_size // (num_folds + 2)) * i
        radius += random.randint(-5, 5)
        if i % 2 == 0:
            fold_color, fold_width = (200, 130, 130), random.randint(fold_width_range[0], fold_width_range[1])
        else:
            fold_color, fold_width = (170, 100, 100), random.randint(fold_width_range[0] + 2, fold_width_range[1] + 2)
        opacity_factor = 0.7 + 0.3 * (i / num_folds)
        adjusted_color = tuple(int(c * opacity_factor) for c in fold_color)
        squish_factor = 0.6 + 0.2 * (i / num_folds)
        ellipse_bounds = [center - radius, center - radius * squish_factor, center + radius, center + radius * squish_factor]
        draw.ellipse(ellipse_bounds, outline=adjusted_color, width=fold_width)
    
    # Dark central area
    central_radius = img_size // (num_folds + 2)
    for r in range(central_radius, 0, -2):
        darkness = 0.6 - (r / central_radius) * 0.4
        dark_color = tuple(int(c * (1-darkness)) for c in (170, 100, 100))
        inner_ellipse = [center - r, center - r * 0.6, center + r, center + r * 0.6]
        draw.ellipse(inner_ellipse, fill=dark_color, outline=None)
    
    return np.array(pil_img) / 255.0

def generate_polyp_texture(mask, base_color):
    img_size = mask.shape[0]
    texture = np.zeros((img_size, img_size, 3))
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0: return texture
    center_y, center_x = int(np.mean(y_indices)), int(np.mean(x_indices))
    max_dist = np.max(np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2))
    y, x = np.ogrid[:img_size, :img_size]
    dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    normalized_dist = np.zeros_like(dist_from_center)
    normalized_dist[mask > 0] = dist_from_center[mask > 0] / max_dist
    highlight_dir = random.uniform(0, 2 * np.pi)
    highlight_x, highlight_y = np.cos(highlight_dir), np.sin(highlight_dir)
    y_rel, x_rel = (y - center_y) / (max_dist + 1e-6), (x - center_x) / (max_dist + 1e-6)
    directional_component = x_rel * highlight_x + y_rel * highlight_y
    for c in range(3):
        color_map = np.ones((img_size, img_size)) * base_color[c]
        color_map[mask > 0] *= (1 - 0.3 * normalized_dist[mask > 0] + 0.2 * directional_component[mask > 0])
        color_map[mask > 0] += np.random.randn(len(y_indices)) * 0.05
        texture[:, :, c] = mask * np.clip(color_map, 0, 1)
    return texture

def generate_polyp(img_size=256, min_radius=10, max_radius=30):
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cx, cy = random.randint(img_size // 4, 3 * img_size // 4), random.randint(img_size // 4, 3 * img_size // 4)
    radius = random.randint(min_radius, max_radius)
    mask_img = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_img)
    squish = random.uniform(0.7, 1.0)
    draw.ellipse([cx - radius, cy - radius * squish, cx + radius, cy + radius * squish], fill=1)
    if random.random() < 0.4:
        ba, bd, bs = random.uniform(0, 2 * np.pi), radius * 0.7, radius * random.uniform(0.3, 0.5)
        bx, by = cx + bd * np.cos(ba), cy + bd * np.sin(ba)
        draw.ellipse([bx - bs, by - bs, bx + bs, by + bs], fill=1)
    mask = np.array(mask_img)
    p_color = np.array([180, 100, 100])/255.0 if random.random() > 0.3 else np.array([220, 120, 120])/255.0
    return mask, generate_polyp_texture(mask, p_color)

def add_polyp_to_image(img, mask, polyp_texture):
    result = img.copy()
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0: return result
    cy, cx = int(np.mean(y_indices)), int(np.mean(x_indices))
    max_dist = np.max(np.sqrt((y_indices - cy)**2 + (x_indices - cx)**2))
    soft_mask = mask.copy().astype(float)
    for y, x in zip(y_indices, x_indices):
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        if dist / max_dist > 0.85:
            soft_mask[y, x] = 1.0 - ((dist / max_dist - 0.85) / 0.15)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - soft_mask) + polyp_texture[:, :, c]
    return np.clip(result, 0, 1)

# =================================================================
# Part 2: Split and Save Logic (Compatible with your 2nd script)
# =================================================================

if __name__ == "__main__":
    args = parse_args()
    random.seed(42)
    np.random.seed(42)

    SPLITS = ["train", "val", "test"]
    CLASSES = ["with_ellipse", "without_ellipse"]
    RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(args.output_dir, split, cls), exist_ok=True)

    counts = {s: int(args.num_images * r) for s, r in RATIOS.items()}
    # Ensure exact total
    counts["test"] = args.num_images - counts["train"] - counts["val"]
    
    datasets_info = {s: [] for s in SPLITS}
    image_id = 0

    # Fold logic scales with image size
    f_width = (max(1, args.image_size // 30), max(2, args.image_size // 15))
    p_rad = (max(2, args.image_size // 25), max(5, args.image_size // 10))

    for split in SPLITS:
        num_per_class = counts[split] // 2
        for has_polyp in [True, False]:
            class_dir = "with_ellipse" if has_polyp else "without_ellipse"
            
            for _ in tqdm(range(num_per_class), desc=f"{split} - {class_dir}"):
                img = generate_colon_background(args.image_size)
                img = add_colon_folds(img, fold_width_range=f_width)
                
                if has_polyp:
                    for _ in range(random.randint(1, 2)):
                        m, t = generate_polyp(args.image_size, min_radius=p_rad[0], max_radius=p_rad[1])
                        img = add_polyp_to_image(img, m, t)
                
                # Convert to PIL and Grayscale for QNN script
                img_uint8 = (img * 255).astype(np.uint8)
                img_obj = Image.fromarray(img_uint8).convert("L")
                
                img_name = f"{image_id}.png"
                img_path = os.path.join(args.output_dir, split, class_dir, img_name)
                img_obj.save(img_path)
                
                datasets_info[split].append({"filename": img_path, "label": 1 if has_polyp else 0})
                image_id += 1

    for split in SPLITS:
        pd.DataFrame(datasets_info[split]).to_csv(os.path.join(args.output_dir, f"labels_{split}.csv"), index=False)

    print(f"\nDone! Original pink-logic dataset created at: {args.output_dir}")