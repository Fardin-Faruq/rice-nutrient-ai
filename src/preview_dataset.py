"""
preview_dataset.py
------------------
Generates a visual quality preview contact sheet from the processed 96x96 dataset.
Selects 6 representative training images from each class across diverse original
aspect ratios to verify leaf tip, edge, and color lesion visibility.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_preview(manifest_csv="outputs/split_manifest.csv", output_png="outputs/dataset_preview.png"):
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / manifest_csv
    out_path = project_root / output_png

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}. Run prepare_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"].copy()

    # Calculate aspect ratio (elongation factor: max_dim / min_dim)
    train_df["max_dim"] = train_df[["original_width", "original_height"]].max(axis=1)
    train_df["min_dim"] = train_df[["original_width", "original_height"]].min(axis=1)
    train_df["aspect_ratio"] = train_df["max_dim"] / train_df["min_dim"]

    classes = ["Nitrogen", "Phosphorus", "Potassium"]
    selected_rows = []

    for cls_name in classes:
        cls_df = train_df[train_df["class_name"] == cls_name].sort_values("aspect_ratio").reset_index(drop=True)
        n = len(cls_df)
        if n == 0:
            print(f"[WARNING] No training images found for class {cls_name}")
            continue

        # Select 6 diverse samples spanning min, 20%, 40%, 60%, 80%, and max aspect ratio
        indices = np.linspace(0, n - 1, 6, dtype=int)
        for idx in indices:
            selected_rows.append(cls_df.iloc[idx])

    # Configure plot (3 rows = classes, 6 columns = aspect ratio progression)
    fig, axes = plt.subplots(3, 6, figsize=(15, 8.5), facecolor="#f8f9fa")
    fig.suptitle(
        "Rice Nutrient AI - 96x96 Processed Preview (Orientation-Aligned Letterbox)\n"
        "6 Training Samples per Class Across Diverse Aspect Ratios",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    for i, row in enumerate(selected_rows):
        r = i // 6
        c = i % 6
        ax = axes[r, c]

        img_path = Path(row["splits_path"])
        if not img_path.exists():
            img_path = Path(row["processed_path"])

        if img_path.exists():
            with Image.open(img_path) as img:
                ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing File", ha="center", va="center")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")

        # Subtitle: original file name and dimensions
        fname = row["file_name"]
        orig_w = row["original_width"]
        orig_h = row["original_height"]
        ar = row["aspect_ratio"]
        ax.set_title(f"{fname}\n{orig_w}x{orig_h} (AR {ar:.1f}:1)", fontsize=8, pad=3)

        # Class label on the far left of each row
        if c == 0:
            ax.set_ylabel(
                row["class_name"],
                fontsize=13,
                fontweight="bold",
                labelpad=10,
                color="#1b4332" if row["class_name"] == "Nitrogen" else (
                    "#7f4f24" if row["class_name"] == "Phosphorus" else "#582f0e"
                )
            )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Contact sheet preview saved to: {out_path.resolve()}")
    print(f"File exists: {out_path.exists()}, File size: {out_path.stat().st_size:,} bytes")

if __name__ == "__main__":
    generate_preview()
