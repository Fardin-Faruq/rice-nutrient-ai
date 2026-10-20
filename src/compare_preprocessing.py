"""
compare_preprocessing.py
------------------------
Visually compares three preprocessing strategies on the same 18 training images
(6 representative aspect ratios per class for Nitrogen, Phosphorus, and Potassium):

Strategy 1: Current orientation-aligned letterboxing (neutral black background).
Strategy 2: Aspect-preserving resize with a neutral light background (RGB 240, 240, 240).
Strategy 3: Saliency-guided leaf-region cropping: detects the most symptomatic/apical
            leaf region with a margin and resizes to 96x96, avoiding destructive center cropping.
            Flags images where foreground/saliency detection has low confidence.

Outputs a 6x9 comparison matrix to outputs/preprocessing_comparison.png.
Does NOT modify existing processed files or split manifests.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Prevent DecompressionBombError on high-res images
Image.MAX_IMAGE_PIXELS = None

TARGET_SIZE = (96, 96)

def align_orientation(img_rgb):
    """Rotates landscape leaves to canonical portrait orientation."""
    w, h = img_rgb.size
    if w > h:
        return img_rgb.rotate(90, expand=True)
    return img_rgb

def letterbox_image(img_rgb, target_size=TARGET_SIZE, bg_color=(0, 0, 0)):
    """Aspect-preserving resize centered on a canvas of specified color."""
    w, h = img_rgb.size
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), bg_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas

def detect_and_crop_leaf_roi(img_rgb, target_size=TARGET_SIZE):
    """
    Detects the most symptomatic/diagnostic leaf region with a margin.
    If background separation or foreground confidence is too low, flags the image.
    """
    arr = np.array(img_rgb)
    h, w, _ = arr.shape

    # 1. Background estimation from perimeter samples
    corners = np.concatenate([
        arr[:15, :15], arr[:15, -15:],
        arr[-15:, :15], arr[-15:, -15:]
    ], axis=0)
    bg_est = corners.mean(axis=(0, 1))

    # Color difference from estimated background
    diff = np.linalg.norm(arr.astype(float) - bg_est, axis=2)
    contrast = diff.mean()

    # 2. Symptom saliency map
    # Healthy rice leaf is predominantly green (G > R and G > B).
    # Deficiency symptoms introduce chlorosis/yellowing (high R+G), bronzing/necrosis (high R, lower G/B),
    # or purpling/darkening (altered R/B balance).
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)
    symptom_map = np.abs(r - g) + np.maximum(0, r - b) + np.abs(r - 128.0) * 0.2
    row_symptoms = symptom_map.mean(axis=1)

    # Flag condition: if row contrast is virtually flat or image is uniform
    status_flag = "OK"
    if contrast < 20 or (row_symptoms.max() - row_symptoms.min()) < 10:
        status_flag = "LOW_CONFIDENCE"

    # 3. Crop window selection: target an aspect ratio ~ 1.5:1 to 2.5:1
    # This prevents the leaf from becoming an ultra-narrow 5px line when scaled to 96x96
    win_h = min(h, max(int(w * 2.0), 300))
    if win_h >= h:
        best_y = 0
        win_h = h
    else:
        # Sliding window along height to find the densest symptom / lesion segment
        win_scores = np.convolve(row_symptoms, np.ones(win_h) / win_h, mode="valid")
        best_y = int(np.argmax(win_scores))

    # Add margin around window if possible
    margin = int(win_h * 0.05)
    y_start = max(0, best_y - margin)
    y_end = min(h, best_y + win_h + margin)

    cropped_arr = arr[y_start:y_end, :, :]
    cropped_img = Image.fromarray(cropped_arr)

    # Resize cropped region with letterbox so it fits target 96x96 without distortion
    final_img = letterbox_image(cropped_img, target_size=target_size, bg_color=(0, 0, 0))
    crop_info = f"y:[{y_start}:{y_end}]"

    return final_img, status_flag, crop_info

def run_comparison(manifest_csv="outputs/split_manifest.csv", output_png="outputs/preprocessing_comparison.png"):
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / manifest_csv
    out_file = project_root / output_png

    if not manifest_path.exists():
        print(f"[ERROR] Missing manifest: {manifest_path}")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"].copy()

    # Calculate aspect ratio
    train_df["max_dim"] = train_df[["original_width", "original_height"]].max(axis=1)
    train_df["min_dim"] = train_df[["original_width", "original_height"]].min(axis=1)
    train_df["aspect_ratio"] = train_df["max_dim"] / train_df["min_dim"]

    classes = ["Nitrogen", "Phosphorus", "Potassium"]
    selected_by_class = {}

    for cls_name in classes:
        cls_df = train_df[train_df["class_name"] == cls_name].sort_values("aspect_ratio").reset_index(drop=True)
        indices = np.linspace(0, len(cls_df) - 1, 6, dtype=int)
        selected_by_class[cls_name] = [cls_df.iloc[i] for i in indices]

    # Create 6x9 figure (6 aspect ratio tiers, 3 classes x 3 strategies)
    fig, axes = plt.subplots(6, 9, figsize=(23, 16), facecolor="#f4f5f7")
    fig.suptitle(
        "Preprocessing Strategies Comparison for Rice Leaf Nutrient Deficiency\n"
        "Columns: [Strategy 1: Black Letterbox] | [Strategy 2: Light Letterbox] | [Strategy 3: Saliency Leaf Crop]\n"
        "Grouped by Class (Nitrogen, Phosphorus, Potassium) across 6 Aspect-Ratio Tiers (Top to Bottom: Low AR to Extreme AR)",
        fontsize=14,
        fontweight="bold",
        y=0.985
    )

    class_col_offsets = {
        "Nitrogen": 0,
        "Phosphorus": 3,
        "Potassium": 6
    }

    # Column titles
    strat_names = ["1. Black Letterbox", "2. Light Letterbox", "3. Leaf-Region Crop"]
    for cls_name, offset in class_col_offsets.items():
        for s_idx, s_name in enumerate(strat_names):
            col_idx = offset + s_idx
            axes[0, col_idx].set_title(
                f"{cls_name}\n{s_name}",
                fontsize=9,
                fontweight="bold",
                pad=8,
                color="#1b4332" if cls_name == "Nitrogen" else ("#7f4f24" if cls_name == "Phosphorus" else "#582f0e")
            )

    # Process all cells
    for cls_name in classes:
        offset = class_col_offsets[cls_name]
        samples = selected_by_class[cls_name]

        for row_idx, sample in enumerate(samples):
            fpath = Path(sample["original_path"])
            fname = sample["file_name"]
            orig_w = sample["original_width"]
            orig_h = sample["original_height"]
            ar = sample["aspect_ratio"]

            with Image.open(fpath) as raw_img:
                img_rgb = align_orientation(raw_img.convert("RGB"))

            # Strategy 1: Current orientation-aligned letterbox (black)
            img_s1 = letterbox_image(img_rgb, target_size=TARGET_SIZE, bg_color=(0, 0, 0))

            # Strategy 2: Orientation-aligned letterbox with light background (240, 240, 240)
            img_s2 = letterbox_image(img_rgb, target_size=TARGET_SIZE, bg_color=(240, 240, 240))

            # Strategy 3: Salient leaf-region crop
            img_s3, flag, crop_info = detect_and_crop_leaf_roi(img_rgb, target_size=TARGET_SIZE)

            # Render S1
            ax1 = axes[row_idx, offset + 0]
            ax1.imshow(img_s1)
            ax1.set_xticks([])
            ax1.set_yticks([])
            if row_idx > 0:
                ax1.set_title(f"{fname}\n{orig_w}x{orig_h} (AR {ar:.1f}:1)", fontsize=7, pad=2)

            # Render S2
            ax2 = axes[row_idx, offset + 1]
            ax2.imshow(img_s2)
            ax2.set_xticks([])
            ax2.set_yticks([])
            if row_idx > 0:
                ax2.set_title(f"{fname}\nLight Pad", fontsize=7, pad=2)

            # Render S3
            ax3 = axes[row_idx, offset + 2]
            ax3.imshow(img_s3)
            ax3.set_xticks([])
            ax3.set_yticks([])
            s3_title = f"{fname}\n{crop_info}"
            if flag != "OK":
                s3_title += f"\n[{flag}]"
                for spine in ax3.spines.values():
                    spine.set_color("red")
                    spine.set_linewidth(2)
            else:
                for spine in ax3.spines.values():
                    spine.set_color("#2a9d8f")
                    spine.set_linewidth(1.5)
            if row_idx > 0:
                ax3.set_title(s3_title, fontsize=7, pad=2)

            # Leftmost tier indicator on row 0 column 0
            if offset == 0:
                axes[row_idx, 0].set_ylabel(f"Tier {row_idx+1}\nAR {ar:.1f}:1", fontsize=9, fontweight="bold", labelpad=5)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Comparison visual saved to: {out_file.resolve()}")
    print(f"File exists: {out_file.exists()}, Size: {out_file.stat().st_size:,} bytes")

if __name__ == "__main__":
    run_comparison()
