import os
import sys
import json
import csv
from pathlib import Path
from collections import Counter
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_class_directories(raw_root, target_classes):
    """
    Recursively scans raw_root for directories matching target class names.
    Handles naming variations such as Nitrogen(N), Phosphorus(P), Potassium(K).
    """
    class_map = {}
    raw_path = Path(raw_root).resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_path}")

    # Look for matching directories anywhere inside raw_path
    for dirpath, dirnames, _ in os.walk(raw_path):
        for dirname in dirnames:
            dir_full = Path(dirpath) / dirname
            dir_lower = dirname.lower()
            for target_cls in target_classes:
                cls_lower = target_cls.lower()
                # Check for exact or prefix/substring match e.g. "nitrogen(n)" or "nitrogen"
                if cls_lower in dir_lower:
                    if target_cls not in class_map:
                        class_map[target_cls] = dir_full

    return class_map

def inspect_dataset(config_path="config.json"):
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / config.get("paths", {}).get("raw", "data/raw")
    outputs_dir = project_root / config.get("paths", {}).get("outputs", "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    classes = config.get("classes", ["Nitrogen", "Phosphorus", "Potassium"])
    if config.get("healthy_class_enabled", False) and "Healthy" not in classes:
        classes.append("Healthy")

    print("=" * 70)
    print("RICE NUTRIENT AI - DATASET AUDIT & INSPECTION")
    print("=" * 70)
    print(f"Project root   : {project_root}")
    print(f"Raw data root  : {raw_dir}")
    print(f"Target classes : {classes}")
    print("-" * 70)

    # 1. Discover class folders
    class_dirs = find_class_directories(raw_dir, classes)

    print("\n[1] Detected Class Folders:")
    for target_cls in classes:
        folder = class_dirs.get(target_cls)
        if folder:
            print(f"  - {target_cls:12s} -> {folder}")
        else:
            print(f"  - {target_cls:12s} -> NOT FOUND")

    missing = [c for c in classes if c not in class_dirs]
    if missing:
        print(f"\n[WARNING] Missing class folder(s): {missing}")

    # 2. Audit files
    audit_records = []
    class_stats = {}

    for target_cls, folder in class_dirs.items():
        all_files = [f for f in folder.iterdir() if f.is_file()]
        supported_files = [
            f for f in all_files if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        non_supported_files = [
            f for f in all_files if f.suffix.lower() not in SUPPORTED_EXTENSIONS
        ]

        valid_count = 0
        corrupt_count = 0
        dimensions = []
        modes = []

        for fpath in supported_files:
            file_size = fpath.stat().st_size
            is_valid = False
            error_msg = ""
            width, height, mode, img_format = None, None, None, None

            try:
                # First pass: verification of headers/structure
                with Image.open(fpath) as img:
                    img.verify()

                # Second pass: decode image pixels to check for corruption/truncation
                with Image.open(fpath) as img:
                    img.load()
                    width, height = img.size
                    mode = img.mode
                    img_format = img.format
                    is_valid = True
                    valid_count += 1
                    dimensions.append((width, height))
                    modes.append(mode)

            except Exception as e:
                is_valid = False
                error_msg = str(e)
                corrupt_count += 1

            audit_records.append({
                "class_name": target_cls,
                "file_name": fpath.name,
                "file_path": str(fpath.resolve()),
                "file_size_bytes": file_size,
                "is_valid": is_valid,
                "width": width if width is not None else "",
                "height": height if height is not None else "",
                "channels": len(mode) if mode else "",
                "mode": mode if mode else "",
                "format": img_format if img_format else "",
                "error": error_msg
            })

        class_stats[target_cls] = {
            "total_files": len(all_files),
            "supported_images": len(supported_files),
            "non_supported_files": len(non_supported_files),
            "valid_images": valid_count,
            "corrupt_images": corrupt_count,
            "dimensions": Counter(dimensions),
            "modes": Counter(modes)
        }

    # 3. Print audit summary
    print("\n[2] Class Summary & Integrity:")
    total_files_all = 0
    total_valid_all = 0
    total_corrupt_all = 0

    for cls_name, stats in class_stats.items():
        total_files_all += stats["total_files"]
        total_valid_all += stats["valid_images"]
        total_corrupt_all += stats["corrupt_images"]
        print(f"\n  Class: {cls_name}")
        print(f"    Total files found      : {stats['total_files']}")
        print(f"    Supported format images: {stats['supported_images']}")
        print(f"    Non-supported files    : {stats['non_supported_files']}")
        print(f"    Valid readable images  : {stats['valid_images']}")
        print(f"    Corrupt / unreadable   : {stats['corrupt_images']}")

        # Dimensions summary
        top_dims = stats["dimensions"].most_common(3)
        dims_str = ", ".join([f"{w}x{h} ({cnt})" for (w, h), cnt in top_dims])
        print(f"    Dimensions (top)       : {dims_str if dims_str else 'N/A'}")

        # Color modes summary
        modes_str = ", ".join([f"{m} ({cnt})" for m, cnt in stats["modes"].items()])
        print(f"    Colour modes           : {modes_str if modes_str else 'N/A'}")

    print("\n" + "-" * 70)
    print(f"OVERALL SUMMARY:")
    print(f"  Total files scanned: {total_files_all}")
    print(f"  Total valid images : {total_valid_all}")
    print(f"  Total corrupt      : {total_corrupt_all}")
    print("-" * 70)

    # 4. Save CSV report
    csv_path = outputs_dir / "dataset_audit.csv"
    fieldnames = [
        "class_name", "file_name", "file_path", "file_size_bytes",
        "is_valid", "width", "height", "channels", "mode", "format", "error"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_records)

    print(f"\n[3] CSV Audit report saved to: {csv_path.resolve()}")
    print("=" * 70)
    return class_stats, csv_path

if __name__ == "__main__":
    inspect_dataset()
