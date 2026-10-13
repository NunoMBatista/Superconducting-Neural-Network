#python3 00c_RealDataset_CVCClinicDB.py ---patch_size 64  --train_count 512 --val_count 126 --test_count 126 --output_dir ./Datasets/dataset_64by64_Train512Val126Test126

import os
import random
import argparse
import shutil
import cv2
import numpy as np
import pandas as pd
import kagglehub
import albumentations as A
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Download, Patch, Augment, and Split CVC-ClinicDB Dataset")
    parser.add_argument("--patch_size", type=int, default=64, help="Size of patches to extract")
    parser.add_argument("--train_count", type=int, default=512, help="Polyp/Non-Polyp count per split for train")
    parser.add_argument("--val_count", type=int, default=126, help="Polyp/Non-Polyp count per split for val")
    parser.add_argument("--test_count", type=int, default=126, help="Polyp/Non-Polyp count per split for test")
    parser.add_argument("--output_dir", default="./Datasets/dataset_64by64_Real", help="Final organized output directory")
    return parser.parse_args()

def get_augmentations():
    return [
        A.Compose([A.HorizontalFlip(p=1.0)]),
        A.Compose([A.VerticalFlip(p=1.0)]),
        A.Compose([A.Rotate(limit=90, p=1.0)]),
        A.Compose([A.Rotate(limit=45, p=1.0)]),
        A.Compose([A.RandomBrightnessContrast(p=1.0)]),
        A.Compose([A.HueSaturationValue(p=1.0)]),
        A.Compose([A.GaussianBlur(blur_limit=(3, 3), p=0.5)]),
        A.Compose([A.GaussNoise(var_limit=(1.0, 10.0), p=0.5)]),
    ]

def augment_to_target(image_pool, target_count, output_dir, prefix):
    """Fills a directory with original images and augments more if short of target_count"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the original images first
    for idx, img in enumerate(image_pool):
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_raw_{idx}.jpg"), img)
        
    current_count = len(image_pool)
    if current_count >= target_count:
        return
        
    # Augment up to target
    augmentations = get_augmentations()
    generated = current_count
    
    with tqdm(total=target_count - current_count, desc=f"Augmenting {prefix}") as pbar:
        while generated < target_count:
            base_img = random.choice(image_pool)
            aug_func = random.choice(augmentations)
            augmented_img = aug_func(image=base_img)['image']
            
            cv2.imwrite(os.path.join(output_dir, f"{prefix}_aug_{generated}.jpg"), augmented_img)
            generated += 1
            pbar.update(1)

def main():
    args = parse_args()
    total_needed = args.train_count + args.val_count + args.test_count
    
    # 1. Download Dataset locally via kagglehub
    print("Checking/Downloading CVC-ClinicDB dataset...")
    download_path = kagglehub.dataset_download("balraj98/cvcclinicdb")
    print(f"Source dataset located at: {download_path}")
    
    original_dir = os.path.join(download_path, 'PNG', 'Original')
    mask_dir = os.path.join(download_path, 'PNG', 'Ground Truth')
    
    # 2. Setup internal collection lists
    polyp_images_pool = []
    bg_images_pool = []
    
    filenames = [f for f in os.listdir(original_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 3. Extract Patches
    print(f"\nExtracting patches of size {args.patch_size}x{args.patch_size} pixels...")
    for filename in tqdm(filenames, desc="Extracting original patches"):
        img = cv2.imread(os.path.join(original_dir, filename))
        mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
            
        # Extract polyp regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            center_x, center_y = x + w // 2, y + h // 2
            x1 = max(0, center_x - args.patch_size // 2)
            y1 = max(0, center_y - args.patch_size // 2)
            
            if x1 + args.patch_size > img.shape[1]: x1 = max(0, img.shape[1] - args.patch_size)
            if y1 + args.patch_size > img.shape[0]: y1 = max(0, img.shape[0] - args.patch_size)
            
            crop = img[y1:y1+args.patch_size, x1:x1+args.patch_size]
            if crop.shape[0] == args.patch_size and crop.shape[1] == args.patch_size:
                polyp_images_pool.append(crop)
                
        # Extract up to 3 background patches per frame
        h_img, w_img = img.shape[:2]
        if h_img > args.patch_size and w_img > args.patch_size:
            bg_count = 0
            for _ in range(200):
                top = np.random.randint(0, h_img - args.patch_size)
                left = np.random.randint(0, w_img - args.patch_size)
                mask_patch = mask[top:top+args.patch_size, left:left+args.patch_size]
                
                if np.count_nonzero(mask_patch) == 0:
                    patch = img[top:top+args.patch_size, left:left+args.patch_size]
                    bg_images_pool.append(patch)
                    bg_count += 1
                    if bg_count >= 3:
                        break

    # 4. Define staging areas and process augmentations
    temp_polyp_dir = "./tmp_patches/polyp"
    temp_bg_dir = "./tmp_patches/no_polyp"
    
    print(f"\nReal polyps found: {len(polyp_images_pool)}. Target required: {total_needed}")
    augment_to_target(polyp_images_pool, total_needed, temp_polyp_dir, "polyp")
    
    print(f"Real backgrounds found: {len(bg_images_pool)}. Target required: {total_needed}")
    augment_to_target(bg_images_pool, total_needed, temp_bg_dir, "bg")
                
    # 5. Organize & Balance Train/Val/Test Splits
    print("\nSplitting and organizing final clean dataset structure...")
    splits = ["train", "val", "test"]
    classes = {"polyp": "with_ellipse", "no_polyp": "without_ellipse"}
    split_counts = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        
    for split in splits:
        for folder in classes.values():
            os.makedirs(os.path.join(args.output_dir, split, folder), exist_ok=True)
            
    csv_datasets = {"train": [], "val": [], "test": []}
    
    for orig_class, target_folder in classes.items():
        src_folder = temp_polyp_dir if orig_class == "polyp" else temp_bg_dir
        available_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(available_files)
        
        # Slicing indexes
        start_idx = 0
        for split in splits:
            count = split_counts[split]
            split_files = available_files[start_idx : start_idx + count]
            start_idx += count
            
            for f_name in split_files:
                src_p = os.path.join(src_folder, f_name)
                dest_p = os.path.join(args.output_dir, split, target_folder, f_name)
                shutil.copy(src_p, dest_p)
                
                relative_csv_path = os.path.join(args.output_dir, split, target_folder, f_name).replace("\\", "/")
                csv_datasets[split].append({"filename": relative_csv_path, "label": label_val if 'label_val' in locals() else (1 if orig_class == "polyp" else 0)})

    # Save balanced CSV files
    print("\nGenerating balanced local CSV files...")
    for split in splits:
        if csv_datasets[split]:
            df = pd.DataFrame(csv_datasets[split])
            df = df.sample(frac=1).reset_index(drop=True)
            df.to_csv(os.path.join(args.output_dir, f"labels_{split}.csv"), index=False)
            
    shutil.rmtree("./tmp_patches")
    print(f"\nExecution successful! Target local dataset structured in: {args.output_dir}")

if __name__ == "__main__":
    main()