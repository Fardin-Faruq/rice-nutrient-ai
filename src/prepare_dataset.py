"""
prepare_dataset.py
------------------
Creates leakage-safe train/validation/test splits for the rice nutrient dataset:
1. Reads config.json, with support for both legacy 3-class baseline and isolated 4-class pipeline.
2. In 4-class mode (healthy_class_enabled: true):
   - Preserves all 1,156 baseline 3-class split assignments and group IDs strictly unchanged.
   - Audits and detects exact and near-duplicates for the 603 Healthy images.
   - Builds connected components with Union-Find to ensure intra-Healthy duplicates never cross splits.
   - Partitions detected Healthy duplicate groups without splitting a group and reports actual counts separately from targets.
   - Composites RGBA, LA, and transparent palette images onto opaque black before converting to RGB.
   - Normalizes image orientation and resizes to 96x96 using aspect-preserving letterboxing.
   - Preflights the baseline manifest and source inventory before writes.
   - Stages all outputs under a unique sibling directory, validates them, then publishes
     outputs_4class_v1/ with one same-filesystem directory rename.
   - Leaves all baseline data and artifacts untouched.
3. In 3-class mode (--force_three_class):
   - Preserves original behavior for reproducing the 3-class baseline.
"""

import os
import sys
import json
import random
import hashlib
import zipfile
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from PIL import Image
import imagehash

# Prevent DecompressionBombError on large images
Image.MAX_IMAGE_PIXELS = None

DEFAULT_HASH_THRESHOLD = 6
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_BASELINE_RECORDS = 1156
EXPECTED_NUTRIENT_FOLDERS = {
    "Nitrogen": "Nitrogen(N)",
    "Phosphorus": "Phosphorus(P)",
    "Potassium": "Potassium(K)"
}
EXPECTED_SPLITS = {"train", "validation", "test"}
EXPECTED_CLASSES = {"Healthy", "Nitrogen", "Phosphorus", "Potassium"}

class DisjointSet:
    """Disjoint-set / Union-Find data structure with path compression."""
    def __init__(self):
        self.parent = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a

def normalize_class_name(raw_name):
    """Normalizes folder and audit names to canonical class names."""
    name_lower = str(raw_name).strip().lower()
    if "healthy" in name_lower:
        return "Healthy"
    if "nitrogen" in name_lower:
        return "Nitrogen"
    if "phosphorus" in name_lower:
        return "Phosphorus"
    if "potassium" in name_lower:
        return "Potassium"
    return str(raw_name).strip()

def compute_sha256(filepath, chunk_size=65536):
    """Computes byte-level SHA-256 hash for exact duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_perceptual_hashes(filepath):
    """
    Computes perceptual hashes with orientation-invariance.
    Canonicalizes orientation (height >= width) and computes 0 and 180 degree phash.
    """
    with Image.open(filepath) as img:
        gray = img.convert("L")
        if gray.width > gray.height:
            gray = gray.rotate(90, expand=True)
        h0 = imagehash.phash(gray)
        h180 = imagehash.phash(gray.rotate(180))
        return h0, h180

def min_phash_distance(h_pair_a, h_pair_b):
    """Returns minimum hamming distance across canonical and 180-degree flip."""
    h0_a, h180_a = h_pair_a
    h0_b, h180_b = h_pair_b
    d1 = h0_a - h0_b
    d2 = h0_a - h180_b
    return min(d1, d2)

def process_leaf_image(input_path, target_size=(96, 96), method="letterbox", align_orientation=True):
    """
    Processes a leaf image into target dimensions (96x96 RGB).

    Robust handling of color modes and transparency:
    - If the image is RGBA (or has alpha channel / transparency), it composites
      the image over a neutral black (0, 0, 0) background so transparent margins
      blend seamlessly into the letterbox background, avoiding uninitialized pixels.
    - Orients elongated leaves to canonical portrait (w <= h) to preserve geometry.
    - Letterboxes onto neutral black (0, 0, 0) canvas of target_size.
    """
    with Image.open(input_path) as img:
        # Normalize transparency-bearing modes before compositing on opaque black.
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            canvas = Image.new("RGBA", img.size, (0, 0, 0, 255))
            canvas.alpha_composite(rgba)
            img_rgb = canvas.convert("RGB")
        else:
            img_rgb = img.convert("RGB")

        w, h = img_rgb.size

        if align_orientation and (w > h):
            # Rotate landscape leaves to portrait
            img_rgb = img_rgb.rotate(90, expand=True)
            w, h = img_rgb.size

        target_w, target_h = target_size

        if method == "letterbox":
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = img_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Paste centered on neutral black background
            canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))
            return canvas, w, h

        elif method == "center_crop":
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            cropped = img_rgb.crop((left, top, left + crop_size, top + crop_size))
            resized = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
            return resized, w, h

        else:
            raise ValueError(f"Unknown method: {method}")

def partition_class_groups(groups, target_counts, seed=42):
    """
    Greedy group partitioning into train, val, test to match target counts
    without splitting any connected component.
    """
    rng = random.Random(seed)
    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)

    # Sort large groups first for better bin packing, then smaller groups
    shuffled_groups.sort(key=lambda g: len(g), reverse=True)

    splits = {"train": [], "validation": [], "test": []}
    counts = {"train": 0, "validation": 0, "test": 0}

    for grp in shuffled_groups:
        grp_len = len(grp)
        deficits = {
            s: target_counts[s] - counts[s]
            for s in ("train", "validation", "test")
        }
        best_split = max(deficits, key=deficits.get)
        splits[best_split].append(grp)
        counts[best_split] += grp_len

    return splits, counts

def create_splits_zip(splits_dir, zip_dest):
    """Creates a deterministic ZIP archive from splits_dir."""
    print(f"\n[Packaging] Creating ZIP archive at: {zip_dest}")
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, directories, files in os.walk(splits_dir):
            directories.sort()
            for file in sorted(files):
                abs_file = Path(root) / file
                rel_path = abs_file.relative_to(splits_dir)
                zf.write(abs_file, rel_path.as_posix())
    print(f"  Archive created successfully: {zip_dest} ({zip_dest.stat().st_size:,} bytes)")

def project_relative_path(path, project_root):
    """Returns a project-relative POSIX path, rejecting paths outside the project."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Dataset path is outside the project root: {resolved}") from exc

def validate_preflight_image(image_path):
    """Fully decodes an image and verifies its compatibility with RGB preprocessing."""
    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError("image has non-positive dimensions")

            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                opaque_canvas = Image.new("RGBA", image.size, (0, 0, 0, 255))
                opaque_canvas.alpha_composite(rgba)
                rgb_image = opaque_canvas.convert("RGB")
            else:
                rgb_image = image.convert("RGB")
            rgb_image.load()
    except Exception as exc:
        raise ValueError(
            f"unreadable, unsupported, or incompatible image {image_path}: {exc}"
        ) from exc

def validate_baseline_manifest(manifest_path, raw_dir, project_root):
    """Validates exact baseline-manifest coverage without changing any assignments."""
    required_columns = {
        "file_name", "original_path", "class_name", "group_id", "split",
        "original_width", "original_height"
    }
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required baseline manifest is missing: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    missing_columns = sorted(required_columns - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Baseline manifest is missing columns: {missing_columns}")
    if len(manifest) != EXPECTED_BASELINE_RECORDS:
        raise ValueError(
            f"Baseline manifest has {len(manifest)} rows; expected "
            f"{EXPECTED_BASELINE_RECORDS}."
        )

    nutrient_inventory = {}
    nested_images = []
    errors = []
    for class_name, folder_name in EXPECTED_NUTRIENT_FOLDERS.items():
        class_dir = raw_dir / "rice_plant_lacks_nutrients" / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Required nutrient source directory is missing: {class_dir}")
        for source in class_dir.rglob("*"):
            if source.is_file() and source.suffix.lower() in SUPPORTED_EXTENSIONS:
                if source.parent != class_dir:
                    nested_images.append(source.resolve())
                    continue
                identity = str(source.resolve()).casefold()
                if identity in nutrient_inventory:
                    raise ValueError(f"Duplicate nutrient source identity on disk: {source}")
                nutrient_inventory[identity] = (source.resolve(), class_name)
    if nested_images:
        errors.extend(
            f"supported nutrient image is in a nested directory: {path}"
            for path in sorted(nested_images)
        )

    manifest_records = []
    manifest_identities = set()
    group_assignments = {}

    for row_number, row in manifest.iterrows():
        row_label = row_number + 2
        try:
            raw_path = row["original_path"]
            if pd.isna(raw_path) or not str(raw_path).strip():
                raise ValueError("original_path is blank")
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = project_root / path
            path = path.resolve()
            identity = str(path).casefold()

            class_name = str(row["class_name"]).strip()
            split_name = str(row["split"]).strip()
            group_id = row["group_id"]
            if pd.isna(group_id) or not str(group_id).strip():
                raise ValueError("group_id is blank")
            if str(group_id).strip() != str(group_id):
                raise ValueError("group_id has leading or trailing whitespace")
            if class_name not in EXPECTED_NUTRIENT_FOLDERS:
                raise ValueError(f"invalid nutrient class {class_name!r}")
            if split_name not in EXPECTED_SPLITS:
                raise ValueError(f"invalid split {split_name!r}")
            if identity in manifest_identities:
                raise ValueError(f"duplicate image identity {path}")
            manifest_identities.add(identity)
            if not path.is_file():
                raise ValueError(f"source image does not exist: {path}")

            inventory_item = nutrient_inventory.get(identity)
            if inventory_item is None:
                raise ValueError(f"source is not in the expected nutrient inventory: {path}")
            if inventory_item[1] != class_name:
                raise ValueError(
                    f"class mismatch for {path}: manifest={class_name}, "
                    f"source folder={inventory_item[1]}"
                )
            if str(row["file_name"]) != path.name:
                raise ValueError(f"file_name does not match original_path for {path}")
            validate_preflight_image(path)

            try:
                width = int(row["original_width"])
                height = int(row["original_height"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid image dimensions for {path}") from exc
            if width <= 0 or height <= 0:
                raise ValueError(f"non-positive image dimensions for {path}")

            group_key = str(group_id)
            previous_assignment = group_assignments.setdefault(
                group_key, (class_name, split_name)
            )
            if previous_assignment != (class_name, split_name):
                raise ValueError(
                    f"group_id {group_id!r} has inconsistent class/split assignments "
                    f"({previous_assignment}, {(class_name, split_name)})"
                )

            manifest_records.append({
                "file_name": str(row["file_name"]),
                "original_path": project_relative_path(path, project_root),
                "class_name": class_name,
                "group_id": group_id,
                "split": split_name,
                "original_width": width,
                "original_height": height
            })
        except (ValueError, TypeError, OSError) as exc:
            errors.append(f"manifest row {row_label}: {exc}")

    inventory_identities = set(nutrient_inventory)
    missing = sorted(inventory_identities - manifest_identities)
    extra = sorted(manifest_identities - inventory_identities)
    if len(nutrient_inventory) != EXPECTED_BASELINE_RECORDS:
        errors.append(
            f"nutrient source inventory has {len(nutrient_inventory)} supported files; "
            f"expected {EXPECTED_BASELINE_RECORDS}"
        )
    if missing:
        errors.append(f"{len(missing)} nutrient source(s) missing from manifest: {missing[:10]}")
    if extra:
        errors.append(f"{len(extra)} manifest source(s) are not expected nutrient files: {extra[:10]}")
    if errors:
        raise ValueError("Baseline preflight failed:\n- " + "\n- ".join(errors))

    return manifest_records

def normalize_report_paths(frame, path_columns, project_root):
    """Converts path fields in copied reports to portable project-relative paths."""
    result = frame.copy()
    for column in path_columns:
        if column not in result.columns:
            continue
        normalized = []
        for value in result[column]:
            if pd.isna(value) or not str(value).strip():
                normalized.append(value)
                continue
            path = Path(str(value))
            if not path.is_absolute():
                path = project_root / path
            normalized.append(project_relative_path(path, project_root))
        result[column] = normalized
    return result

def validate_image_tree(base_dir, expected_records, path_column, root_name):
    """Validates class folders, exact file coverage, and image readability."""
    expected_directories = {
        Path(split)
        for split in EXPECTED_SPLITS
    } | {
        Path(split) / class_name
        for split in EXPECTED_SPLITS
        for class_name in EXPECTED_CLASSES
    }
    actual_directories = {
        path.relative_to(base_dir)
        for path in base_dir.rglob("*")
        if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise ValueError(
            f"Unexpected directory structure under {base_dir}: "
            f"missing={sorted(expected_directories - actual_directories)}, "
            f"extra={sorted(actual_directories - expected_directories)}"
        )

    expected_paths = set()
    for record in expected_records:
        relative_path = Path(record[path_column])
        try:
            relative_path = relative_path.relative_to(root_name)
        except ValueError as exc:
            raise ValueError(
                f"{path_column} is not rooted at {root_name!r}: {relative_path}"
            ) from exc
        expected_paths.add(relative_path.as_posix())
    if root_name == "splits":
        expected_paths.update({"split_manifest.csv", "split_summary.csv"})
    actual_files = {
        path.relative_to(base_dir).as_posix()
        for path in base_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_paths:
        raise ValueError(
            f"Image coverage mismatch under {base_dir}: "
            f"missing={sorted(expected_paths - actual_files)[:10]}, "
            f"extra={sorted(actual_files - expected_paths)[:10]}"
        )

    for relative_path in sorted(expected_paths):
        image_path = base_dir / Path(relative_path)
        if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            with Image.open(image_path) as image:
                image.verify()

def validate_staged_outputs(stage_root, manifest_records, baseline_records, healthy_components):
    """Validates every staged artifact before the output-root rename."""
    expected_root_entries = {"splits", "processed", "reports", "rice_splits_4class.zip"}
    actual_root_entries = {path.name for path in stage_root.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise ValueError(
            f"Unexpected staged root entries: "
            f"missing={sorted(expected_root_entries - actual_root_entries)}, "
            f"extra={sorted(actual_root_entries - expected_root_entries)}"
        )

    splits_dir = stage_root / "splits"
    processed_dir = stage_root / "processed"
    reports_dir = stage_root / "reports"
    validate_image_tree(splits_dir, manifest_records, "splits_path", "splits")
    validate_image_tree(processed_dir, manifest_records, "processed_path", "processed")

    expected_report_files = {
        "split_manifest.csv", "split_summary.csv", "dataset_audit.csv",
        "exact_duplicates.csv", "near_duplicate_review.csv"
    }
    actual_report_entries = {path.name for path in reports_dir.iterdir()}
    if actual_report_entries != expected_report_files or any(
        not path.is_file() for path in reports_dir.iterdir()
    ):
        raise ValueError(
            f"Unexpected staged report files: "
            f"missing={sorted(expected_report_files - actual_report_entries)}, "
            f"extra={sorted(actual_report_entries - expected_report_files)}"
        )

    split_manifest_path = splits_dir / "split_manifest.csv"
    report_manifest_path = reports_dir / "split_manifest.csv"
    manifest_df = pd.read_csv(split_manifest_path)
    report_manifest_df = pd.read_csv(report_manifest_path)
    pd.testing.assert_frame_equal(manifest_df, report_manifest_df, check_dtype=False)

    required_columns = {
        "file_name", "original_path", "splits_path", "processed_path",
        "class_name", "group_id", "split", "original_width", "original_height",
        "processed_width", "processed_height"
    }
    if not required_columns.issubset(manifest_df.columns):
        raise ValueError("Staged split manifest is missing required columns.")
    if len(manifest_df) != len(manifest_records):
        raise ValueError(
            f"Staged manifest has {len(manifest_df)} records; "
            f"expected {len(manifest_records)}."
        )
    if manifest_df["original_path"].duplicated().any():
        raise ValueError("Staged manifest contains duplicate source identities.")

    expected_rows = {
        (
            record["file_name"], record["original_path"], record["class_name"],
            str(record["group_id"]), record["split"], record["splits_path"],
            record["processed_path"], record["original_width"],
            record["original_height"], record["processed_width"],
            record["processed_height"]
        )
        for record in manifest_records
    }
    actual_rows = {
        (
            row.file_name, row.original_path, row.class_name, str(row.group_id),
            row.split, row.splits_path, row.processed_path, row.original_width,
            row.original_height, row.processed_width, row.processed_height
        )
        for row in manifest_df.itertuples(index=False)
    }
    if actual_rows != expected_rows:
        raise ValueError("Staged manifest rows do not match generated records.")

    for path_column in ("original_path", "splits_path", "processed_path"):
        for value in manifest_df[path_column].astype(str):
            relative = Path(value)
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                raise ValueError(f"Manifest contains a non-portable path: {value}")
    for row in manifest_df.itertuples(index=False):
        for relative_value in (row.splits_path, row.processed_path):
            resolved = (stage_root / Path(relative_value)).resolve()
            if not resolved.is_relative_to(stage_root.resolve()) or not resolved.is_file():
                raise ValueError(f"Manifest output path is invalid or missing: {relative_value}")

    for report_name in expected_report_files:
        report_df = pd.read_csv(reports_dir / report_name)
        path_columns = {
            "file_path", "file_a_path", "file_b_path"
        }.intersection(report_df.columns)
        for column in path_columns:
            for value in report_df[column].dropna().astype(str):
                relative = Path(value)
                if relative.is_absolute() or relative.drive or ".." in relative.parts:
                    raise ValueError(
                        f"Report {report_name} contains a non-portable path: {value}"
                    )

    baseline_rows = {
        (
            record["original_path"], record["class_name"], str(record["group_id"]),
            record["split"]
        )
        for record in baseline_records
    }
    staged_baseline_rows = {
        (
            row.original_path, row.class_name, str(row.group_id), row.split
        )
        for row in manifest_df.itertuples(index=False)
        if row.class_name in EXPECTED_NUTRIENT_FOLDERS
    }
    if staged_baseline_rows != baseline_rows:
        raise ValueError("Staged manifest changed baseline class, group, split, or source identity.")

    healthy_rows = manifest_df[manifest_df["class_name"] == "Healthy"]
    healthy_split_by_source = dict(zip(healthy_rows["original_path"], healthy_rows["split"]))
    for component in healthy_components:
        component_splits = {
            healthy_split_by_source.get(project_relative_path(path, stage_root.parent))
            for path in component
        }
        if None in component_splits or len(component_splits) != 1:
            raise ValueError("A detected Healthy duplicate group crosses splits or is missing.")

    split_files = {
        path.relative_to(splits_dir).as_posix()
        for path in splits_dir.rglob("*")
        if path.is_file()
    }
    zip_path = stage_root / "rice_splits_4class.zip"
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Four-class ZIP contains duplicate entry names.")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("Four-class ZIP contains an unsafe entry path.")
        if set(names) != split_files:
            raise ValueError(
                "Four-class ZIP inventory does not match staged splits: "
                f"missing={sorted(split_files - set(names))[:10]}, "
                f"extra={sorted(set(names) - split_files)[:10]}"
            )
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Four-class ZIP integrity check failed at {bad_member}.")
        zipped_manifest = pd.read_csv(archive.open("split_manifest.csv"))
        pd.testing.assert_frame_equal(manifest_df, zipped_manifest, check_dtype=False)
        for member in names:
            if Path(member).suffix.lower() in SUPPORTED_EXTENSIONS:
                with archive.open(member) as image_file:
                    with Image.open(image_file) as image:
                        image.verify()

def build_four_class_staging(
    stage_root, manifest_records, target_classes,
    baseline_audit_df, healthy_audit, baseline_exact_df, healthy_exact,
    baseline_near_df, healthy_near, config, method
):
    """Writes all four-class deliverables exclusively beneath the staging root."""
    target_w = config.get("image_width", 96)
    target_h = config.get("image_height", 96)
    splits_dir = stage_root / "splits"
    processed_dir = stage_root / "processed"
    reports_dir = stage_root / "reports"

    for base_dir in (splits_dir, processed_dir):
        for split_name in EXPECTED_SPLITS:
            for class_name in target_classes:
                (base_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir()

    rows = []
    for record in manifest_records:
        src_path = Path(record["original_path"])
        if not src_path.is_absolute():
            src_path = stage_root.parent / src_path
        cls_name = record["class_name"]
        split_name = record["split"]
        processed_image, original_width, original_height = process_leaf_image(
            src_path,
            target_size=(target_w, target_h),
            method=method,
            align_orientation=True
        )
        output_name = f"{src_path.stem}.jpg"
        split_destination = splits_dir / split_name / cls_name / output_name
        processed_destination = processed_dir / split_name / cls_name / output_name
        processed_image.save(split_destination, "JPEG", quality=95)
        processed_image.save(processed_destination, "JPEG", quality=95)
        rows.append({
            "file_name": record["file_name"],
            "original_path": record["original_path"],
            "splits_path": split_destination.relative_to(stage_root).as_posix(),
            "processed_path": processed_destination.relative_to(stage_root).as_posix(),
            "class_name": cls_name,
            "group_id": record["group_id"],
            "split": split_name,
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": target_w,
            "processed_height": target_h
        })

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(splits_dir / "split_manifest.csv", index=False)
    manifest_df.to_csv(reports_dir / "split_manifest.csv", index=False)

    summary_data = []
    healthy_total = sum(record["class_name"] == "Healthy" for record in rows)
    healthy_targets = {
        "target_train_count": int(round(healthy_total * config.get("train_ratio", 0.70))),
        "target_val_count": int(round(healthy_total * config.get("validation_ratio", 0.15))),
    }
    healthy_targets["target_test_count"] = (
        healthy_total
        - healthy_targets["target_train_count"]
        - healthy_targets["target_val_count"]
    )
    for class_name in target_classes:
        class_records = [record for record in rows if record["class_name"] == class_name]
        actual_counts = Counter(record["split"] for record in class_records)
        class_total = len(class_records)
        targets = healthy_targets if class_name == "Healthy" else {
            "target_train_count": "",
            "target_val_count": "",
            "target_test_count": ""
        }
        summary_data.append({
            "class_name": class_name,
            "total_images": class_total,
            "train_count": actual_counts["train"],
            "val_count": actual_counts["validation"],
            "test_count": actual_counts["test"],
            "train_percent": round(actual_counts["train"] / class_total * 100, 2) if class_total else 0.0,
            "val_percent": round(actual_counts["validation"] / class_total * 100, 2) if class_total else 0.0,
            "test_percent": round(actual_counts["test"] / class_total * 100, 2) if class_total else 0.0,
            **targets
        })
    total_counts = Counter(record["split"] for record in rows)
    total = len(rows)
    summary_data.append({
        "class_name": "TOTAL",
        "total_images": total,
        "train_count": total_counts["train"],
        "val_count": total_counts["validation"],
        "test_count": total_counts["test"],
        "train_percent": round(total_counts["train"] / total * 100, 2) if total else 0.0,
        "val_percent": round(total_counts["validation"] / total * 100, 2) if total else 0.0,
        "test_percent": round(total_counts["test"] / total * 100, 2) if total else 0.0,
        "target_train_count": "",
        "target_val_count": "",
        "target_test_count": ""
    })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(splits_dir / "split_summary.csv", index=False)
    summary_df.to_csv(reports_dir / "split_summary.csv", index=False)

    pd.concat([baseline_audit_df, pd.DataFrame(healthy_audit)], ignore_index=True).to_csv(
        reports_dir / "dataset_audit.csv", index=False
    )
    pd.concat([baseline_exact_df, pd.DataFrame(healthy_exact)], ignore_index=True).to_csv(
        reports_dir / "exact_duplicates.csv", index=False
    )
    pd.concat([baseline_near_df, pd.DataFrame(healthy_near)], ignore_index=True).to_csv(
        reports_dir / "near_duplicate_review.csv", index=False
    )

    create_splits_zip(splits_dir, stage_root / "rice_splits_4class.zip")
    return rows, summary_data

def audit_and_group_healthy(healthy_dir, threshold=DEFAULT_HASH_THRESHOLD):
    """
    Audits the Healthy images and detects intra-class exact and near duplicates.
    Returns:
    - audit_records: list of dicts with file metadata
    - exact_records: list of dicts for exact duplicates
    - near_records: list of dicts for near duplicates
    - components: list of lists representing duplicate-connected groups of file paths
    """
    healthy_path = Path(healthy_dir).resolve()
    all_files = sorted([f for f in healthy_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"[Healthy Audit] Found {len(all_files)} supported images in {healthy_path}")

    audit_records = []
    file_record_map = {}
    valid_files = []

    for fpath in all_files:
        file_size = fpath.stat().st_size
        is_valid = False
        error_msg = ""
        width, height, mode, img_format = None, None, None, None

        try:
            with Image.open(fpath) as img:
                img.verify()
            with Image.open(fpath) as img:
                img.load()
                width, height = img.size
                mode = img.mode
                img_format = img.format
                is_valid = True
        except Exception as e:
            is_valid = False
            error_msg = str(e)

        rec = {
            "class_name": "Healthy",
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
        }
        audit_records.append(rec)
        if is_valid:
            valid_files.append(fpath)
            file_record_map[str(fpath.resolve())] = rec

    print(f"[Healthy Audit] Valid: {len(valid_files)} / {len(all_files)}, Corrupted: {len(all_files) - len(valid_files)}")
    if not valid_files or len(valid_files) != len(all_files):
        raise ValueError(
            f"Healthy image audit found {len(all_files) - len(valid_files)} unreadable "
            "image(s); refusing to prepare an incomplete class."
        )

    # 1. Exact duplicates (SHA-256)
    sha_map = defaultdict(list)
    for fpath in valid_files:
        sha = compute_sha256(fpath)
        sha_map[sha].append(fpath)

    exact_records = []
    exact_group_idx = 1
    ds = DisjointSet()

    for sha, group in sha_map.items():
        if len(group) > 1:
            grp_id = f"EXACT_H_{exact_group_idx:03d}"
            exact_group_idx += 1
            for fpath in group:
                exact_records.append({
                    "group_id": grp_id,
                    "sha256": sha,
                    "is_cross_class": False,
                    "class_name": "Healthy",
                    "file_name": fpath.name,
                    "file_path": str(fpath.resolve()),
                    "file_size_bytes": fpath.stat().st_size
                })
            for i in range(len(group) - 1):
                ds.union(str(group[i].resolve()), str(group[i + 1].resolve()))

    print(f"[Healthy Duplicates] Exact duplicate groups: {len(exact_records) // 2 if exact_records else 0} ({len(exact_records)} files)")

    # 2. Near duplicates (pHash)
    print(f"[Healthy Near-Duplicates] Computing perceptual hashes (threshold <= {threshold})...")
    hashes = {}
    for fpath in valid_files:
        hashes[str(fpath.resolve())] = compute_perceptual_hashes(fpath)

    near_records = []
    n = len(valid_files)
    near_pairs = []

    for i in range(n):
        fa = valid_files[i]
        fa_str = str(fa.resolve())
        ha = hashes[fa_str]
        for j in range(i + 1, n):
            fb = valid_files[j]
            fb_str = str(fb.resolve())
            hb = hashes[fb_str]
            dist = min_phash_distance(ha, hb)

            if dist <= threshold:
                ds.union(fa_str, fb_str)
                near_pairs.append((fa, fb, dist))

    print(f"[Healthy Near-Duplicates] Candidate pairs within distance <= {threshold}: {len(near_pairs)}")

    # Group connected components for Healthy
    component_map = defaultdict(list)
    for fpath in valid_files:
        root = ds.find(str(fpath.resolve()))
        component_map[root].append(str(fpath.resolve()))

    # Build near duplicate review records with assigned cluster group_id
    cluster_id_lookup = {}
    multi_clusters = {k: v for k, v in component_map.items() if len(v) > 1}
    for c_idx, (root, members) in enumerate(multi_clusters.items(), 1):
        c_id = f"NEAR_GRP_H_{c_idx:03d}"
        for m in members:
            cluster_id_lookup[m] = c_id

    for fa, fb, dist in near_pairs:
        fa_str = str(fa.resolve())
        fb_str = str(fb.resolve())
        grp_id = cluster_id_lookup.get(fa_str, "UNGROUPED")
        is_exact = compute_sha256(fa) == compute_sha256(fb)
        near_records.append({
            "group_id": grp_id,
            "hash_distance": dist,
            "review_flag": "EXACT_DUPLICATE_SAME_CLASS" if is_exact else "NEAR_DUPLICATE_SAME_CLASS",
            "is_cross_class": False,
            "is_exact_sha256": is_exact,
            "file_a_class": "Healthy",
            "file_a_name": fa.name,
            "file_a_path": fa_str,
            "file_b_class": "Healthy",
            "file_b_name": fb.name,
            "file_b_path": fb_str
        })

    components = list(component_map.values())
    print(f"[Healthy Partition Groups] Total components: {len(components)} ({len(multi_clusters)} duplicate clusters, {len(components) - len(multi_clusters)} singletons)")
    return audit_records, exact_records, near_records, components, file_record_map

def prepare_four_class_dataset(config, project_root, method="letterbox"):
    """
    Builds and validates all four-class artifacts in staging, then publishes once.
    """
    target_w = config.get("image_width", 96)
    target_h = config.get("image_height", 96)
    random_seed = config.get("random_seed", 42)
    train_ratio = config.get("train_ratio", 0.70)
    val_ratio = config.get("validation_ratio", 0.15)
    test_ratio = config.get("test_ratio", 0.15)
    raw_dir = project_root / config.get("paths", {}).get("raw", "data/raw")
    project_root = project_root.resolve()
    configured_root = Path(
        config.get("paths", {}).get("four_class_output_root", "outputs_4class_v1")
    )
    if configured_root.is_absolute():
        raise ValueError("four_class_output_root must be a project-relative path.")
    final_root = project_root / configured_root
    if final_root.parent != project_root or final_root.name != "outputs_4class_v1":
        raise ValueError(
            "four_class_output_root must be the new project-root sibling "
            "'outputs_4class_v1' to protect existing artifacts."
        )

    protected_paths = [
        project_root / "data",
        project_root / "models",
        project_root / config.get("paths", {}).get("outputs", "outputs"),
        project_root / "rice_splits.zip"
    ]
    for protected_path in protected_paths:
        protected_path = protected_path.resolve()
        if (
            final_root == protected_path
            or final_root in protected_path.parents
            or protected_path in final_root.parents
        ):
            raise ValueError(
                f"Four-class output root overlaps protected path: {protected_path}"
            )
    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError(
            f"Four-class output root already exists; refusing to overwrite: {final_root}"
        )

    baseline_outputs_dir = project_root / config.get("paths", {}).get("outputs", "outputs")
    baseline_manifest_path = baseline_outputs_dir / "split_manifest.csv"
    baseline_audit_path = baseline_outputs_dir / "dataset_audit.csv"
    baseline_exact_path = baseline_outputs_dir / "exact_duplicates.csv"
    baseline_near_path = baseline_outputs_dir / "near_duplicate_review.csv"

    print("=" * 80)
    print("RICE NUTRIENT AI - STAGED 4-CLASS DATASET PREPARATION")
    print("=" * 80)
    print(f"Target Dimensions   : {target_w}x{target_h} RGB")
    print(f"Resizing Strategy   : {method} (orientation-aligned)")
    print(f"Target Split Ratios : Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")
    print(f"Random Seed         : {random_seed}")
    print(f"Final Output Root   : {final_root}")
    print("-" * 80)

    if target_w <= 0 or target_h <= 0:
        raise ValueError("Configured image dimensions must be positive.")
    if (
        not all(0.0 <= ratio <= 1.0 for ratio in (train_ratio, val_ratio, test_ratio))
        or abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9
    ):
        raise ValueError("Train, validation, and test ratios must be in [0, 1] and sum to 1.")

    # This read-only preflight completes before staging or dataset output writes.
    baseline_records = validate_baseline_manifest(
        baseline_manifest_path, raw_dir, project_root
    )
    print(f"[Preflight] Validated {len(baseline_records)} unchanged nutrient records.")

    healthy_dir = raw_dir / "rice_plant_lacks_nutrients" / "Healthy(H)"
    if not healthy_dir.exists():
        raise FileNotFoundError(f"Healthy raw directory not found: {healthy_dir}")

    print(f"\n[Step 2] Auditing and grouping Healthy images in: {healthy_dir}")
    h_audit, h_exact, h_near, h_components, h_rec_map = audit_and_group_healthy(healthy_dir, threshold=DEFAULT_HASH_THRESHOLD)
    h_total = len(h_rec_map)
    if h_total == 0:
        raise ValueError("No valid Healthy images were found.")
    target_train_h = int(round(h_total * train_ratio))
    target_val_h = int(round(h_total * val_ratio))
    target_test_h = h_total - target_train_h - target_val_h
    h_targets = {"train": target_train_h, "validation": target_val_h, "test": target_test_h}
    h_splits, h_actual_counts = partition_class_groups(h_components, h_targets, seed=random_seed)
    healthy_split_assignments = {}
    file_to_h_group = {}
    h_group_idx = 1
    for root_members in h_components:
        if len(root_members) > 1:
            grp_id = f"GRP_DUP_H_{h_group_idx:03d}"
            h_group_idx += 1
        else:
            grp_id = f"GRP_SGL_H_{h_group_idx:04d}"
            h_group_idx += 1
        for m in root_members:
            file_to_h_group[m] = grp_id
    for s_name, grp_list in h_splits.items():
        for g in grp_list:
            for fpath in g:
                healthy_split_assignments[fpath] = s_name

    if set(healthy_split_assignments) != set(h_rec_map):
        raise ValueError("Healthy partitioning did not assign every valid image exactly once.")
    for component in h_components:
        component_splits = {healthy_split_assignments[path] for path in component}
        if len(component_splits) != 1:
            raise ValueError("A detected Healthy duplicate group crosses split boundaries.")

    actual_h_counts = Counter(healthy_split_assignments.values())
    if any(actual_h_counts[split] != h_actual_counts[split] for split in EXPECTED_SPLITS):
        raise ValueError("Healthy split counts do not match the assigned image records.")
    train_h = actual_h_counts["train"]
    val_h = actual_h_counts["validation"]
    test_h = actual_h_counts["test"]
    print("\n[Healthy split counts]")
    print(f"  Targets: train={target_train_h}, validation={target_val_h}, test={target_test_h}")
    print(f"  Actual : train={train_h}, validation={val_h}, test={test_h}")

    target_classes = ["Healthy", "Nitrogen", "Phosphorus", "Potassium"]
    manifest_records = list(baseline_records)
    for path_string, record in h_rec_map.items():
        manifest_records.append({
            "file_name": record["file_name"],
            "original_path": project_relative_path(path_string, project_root),
            "class_name": "Healthy",
            "group_id": file_to_h_group[path_string],
            "split": healthy_split_assignments[path_string],
            "original_width": int(record["width"]),
            "original_height": int(record["height"])
        })

    output_identities = set()
    for record in manifest_records:
        output_identity = (
            record["split"], record["class_name"],
            Path(record["original_path"]).stem.casefold()
        )
        if output_identity in output_identities:
            raise ValueError(
                f"Output filename collision would overwrite an image: {output_identity}"
            )
        output_identities.add(output_identity)

    baseline_audit_df = (
        pd.read_csv(baseline_audit_path) if baseline_audit_path.is_file()
        else pd.DataFrame()
    )
    baseline_exact_df = (
        pd.read_csv(baseline_exact_path) if baseline_exact_path.is_file()
        else pd.DataFrame()
    )
    baseline_near_df = (
        pd.read_csv(baseline_near_path) if baseline_near_path.is_file()
        else pd.DataFrame()
    )
    baseline_audit_df = normalize_report_paths(
        baseline_audit_df, ["file_path"], project_root
    )
    h_audit_df = normalize_report_paths(
        pd.DataFrame(h_audit), ["file_path"], project_root
    )
    baseline_exact_df = normalize_report_paths(
        baseline_exact_df, ["file_path"], project_root
    )
    h_exact_df = normalize_report_paths(
        pd.DataFrame(h_exact), ["file_path"], project_root
    )
    baseline_near_df = normalize_report_paths(
        baseline_near_df, ["file_a_path", "file_b_path"], project_root
    )
    h_near_df = normalize_report_paths(
        pd.DataFrame(h_near), ["file_a_path", "file_b_path"], project_root
    )

    staging_root = Path(tempfile.mkdtemp(
        prefix=f".{final_root.name}.staging-",
        dir=str(final_root.parent)
    ))
    print(f"[Staging] Building all deliverables under: {staging_root}")
    try:
        generated_records, summary_data = build_four_class_staging(
            staging_root, manifest_records, target_classes,
            baseline_audit_df, h_audit_df.to_dict("records"),
            baseline_exact_df, h_exact_df.to_dict("records"),
            baseline_near_df, h_near_df.to_dict("records"),
            config, method
        )
        validate_staged_outputs(
            staging_root, generated_records, baseline_records, h_components
        )
    except Exception as exc:
        print(
            f"[FAILED] Final root remains untouched: {final_root}. "
            f"Staging was preserved for inspection: {staging_root}"
        )
        raise RuntimeError(
            f"Four-class staging failed; final root is untouched and staging remains at "
            f"{staging_root}: {exc}"
        ) from exc

    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError(
            f"Final output root appeared before publication; refusing to overwrite it. "
            f"Final root: {final_root}; completed staging remains at {staging_root}"
        )
    try:
        os.rename(staging_root, final_root)
    except OSError as exc:
        print(
            f"[PUBLICATION FAILED] Final root exists={final_root.exists() or final_root.is_symlink()}; "
            f"staging exists={staging_root.exists()}. No cleanup or rollback was attempted."
        )
        raise RuntimeError(
            f"Could not publish {staging_root} as {final_root}; existing data was not removed."
        ) from exc

    print("\n" + "=" * 80)
    print("4-CLASS DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Total Images Processed : {len(generated_records)}")
    print(f"Published Output Root  : {final_root}")
    print(f"Actual split summary   : {summary_data[-1]}")
    print("=" * 80)

def prepare_three_class_legacy(config, project_root, method="letterbox"):
    """Legacy 3-class dataset preparation routine."""
    target_w = config.get("image_width", 96)
    target_h = config.get("image_height", 96)
    target_size = (target_w, target_h)
    random_seed = config.get("random_seed", 42)
    train_ratio = config.get("train_ratio", 0.70)
    val_ratio = config.get("validation_ratio", 0.15)
    test_ratio = config.get("test_ratio", 0.15)

    processed_dir = project_root / config.get("paths", {}).get("processed", "data/processed")
    splits_dir = project_root / config.get("paths", {}).get("splits", "data/splits")
    outputs_dir = project_root / config.get("paths", {}).get("outputs", "outputs")

    audit_csv = outputs_dir / "dataset_audit.csv"
    near_dup_csv = outputs_dir / "near_duplicate_review.csv"
    exact_dup_csv = outputs_dir / "exact_duplicates.csv"

    if not audit_csv.exists():
        print(f"[ERROR] Missing dataset audit: {audit_csv}.")
        sys.exit(1)

    audit_df = pd.read_csv(audit_csv)
    valid_df = audit_df[audit_df["is_valid"] == True].copy() if "is_valid" in audit_df.columns else audit_df.copy()
    valid_df["canonical_class"] = valid_df["class_name"].apply(normalize_class_name)

    file_record_map = {}
    for _, row in valid_df.iterrows():
        p = str(Path(row["file_path"]).resolve())
        file_record_map[p] = {
            "file_path": p,
            "file_name": row["file_name"],
            "class_name": row["canonical_class"],
            "width": int(row.get("width", 0)),
            "height": int(row.get("height", 0))
        }

    total_images = len(file_record_map)
    ds = DisjointSet()
    near_dup_pairs = []

    if near_dup_csv.exists():
        near_df = pd.read_csv(near_dup_csv)
        for _, row in near_df.iterrows():
            fa = str(Path(row["file_a_path"]).resolve())
            fb = str(Path(row["file_b_path"]).resolve())
            if fa in file_record_map and fb in file_record_map:
                ds.union(fa, fb)
                near_dup_pairs.append((fa, fb))

    if exact_dup_csv.exists():
        exact_df = pd.read_csv(exact_dup_csv)
        if not exact_df.empty:
            for _, grp in exact_df.groupby("group_id"):
                paths = [str(Path(p).resolve()) for p in grp["file_path"] if str(Path(p).resolve()) in file_record_map]
                for i in range(len(paths) - 1):
                    ds.union(paths[i], paths[i + 1])

    components = defaultdict(list)
    for fpath in file_record_map:
        root = ds.find(fpath)
        components[root].append(fpath)

    file_to_group = {}
    group_idx = 1
    for root, members in components.items():
        grp_id = f"GRP_DUP_{group_idx:03d}" if len(members) > 1 else f"GRP_SGL_{group_idx:04d}"
        group_idx += 1
        for m in members:
            file_to_group[m] = grp_id

    class_groups = defaultdict(list)
    for root, members in components.items():
        cls_name = set(file_record_map[m]["class_name"] for m in members).pop()
        class_groups[cls_name].append(members)

    split_assignments = {}
    split_summary_data = []

    for cls_name in sorted(class_groups.keys()):
        grps = class_groups[cls_name]
        cls_total = sum(len(g) for g in grps)
        target_train = int(round(cls_total * train_ratio))
        target_val = int(round(cls_total * val_ratio))
        target_test = cls_total - target_train - target_val

        targets = {"train": target_train, "validation": target_val, "test": target_test}
        class_splits, actual_counts = partition_class_groups(grps, targets, seed=random_seed)

        for s_name, grp_list in class_splits.items():
            for g in grp_list:
                for fpath in g:
                    split_assignments[fpath] = s_name

        train_c = actual_counts["train"]
        val_c = actual_counts["validation"]
        test_c = actual_counts["test"]
        split_summary_data.append({
            "class_name": cls_name,
            "total_images": cls_total,
            "train_count": train_c,
            "val_count": val_c,
            "test_count": test_c,
            "train_percent": round(train_c / cls_total * 100, 2),
            "val_percent": round(val_c / cls_total * 100, 2),
            "test_percent": round(test_c / cls_total * 100, 2)
        })

    tot_all = sum(r["total_images"] for r in split_summary_data)
    tot_tr = sum(r["train_count"] for r in split_summary_data)
    tot_va = sum(r["val_count"] for r in split_summary_data)
    tot_te = sum(r["test_count"] for r in split_summary_data)
    split_summary_data.append({
        "class_name": "TOTAL",
        "total_images": tot_all,
        "train_count": tot_tr,
        "val_count": tot_va,
        "test_count": tot_te,
        "train_percent": round(tot_tr / tot_all * 100, 2),
        "val_percent": round(tot_va / tot_all * 100, 2),
        "test_percent": round(tot_te / tot_all * 100, 2)
    })

    for base_d in (splits_dir, processed_dir):
        for s_name in ("train", "validation", "test"):
            for cls_name in class_groups.keys():
                (base_d / s_name / cls_name).mkdir(parents=True, exist_ok=True)

    manifest_records = []
    for idx, (fpath, rec) in enumerate(file_record_map.items(), 1):
        split_name = split_assignments[fpath]
        cls_name = rec["class_name"]
        grp_id = file_to_group[fpath]
        src_path = Path(fpath)

        proc_img, orig_w, orig_h = process_leaf_image(
            src_path, target_size=target_size, method=method, align_orientation=True
        )

        out_fname = f"{src_path.stem}.jpg"
        split_dest = splits_dir / split_name / cls_name / out_fname
        proc_dest = processed_dir / split_name / cls_name / out_fname

        proc_img.save(split_dest, "JPEG", quality=95)
        proc_img.save(proc_dest, "JPEG", quality=95)

        manifest_records.append({
            "file_name": src_path.name,
            "original_path": str(src_path.resolve()),
            "splits_path": str(split_dest.resolve()),
            "processed_path": str(proc_dest.resolve()),
            "class_name": cls_name,
            "group_id": grp_id,
            "split": split_name,
            "original_width": orig_w,
            "original_height": orig_h,
            "processed_width": target_w,
            "processed_height": target_h
        })

    pd.DataFrame(manifest_records).to_csv(outputs_dir / "split_manifest.csv", index=False)
    pd.DataFrame(split_summary_data).to_csv(outputs_dir / "split_summary.csv", index=False)
    print("[Legacy 3-Class Complete]")

def main():
    parser = argparse.ArgumentParser(description="Create leakage-safe train/val/test splits.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--method", default="letterbox", choices=["letterbox", "center_crop"],
                        help="Resizing method: 'letterbox' (recommended for full leaf) or 'center_crop'")
    parser.add_argument("--force_three_class", action="store_true",
                        help="Force legacy 3-class preparation into default baseline paths")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_file = project_root / args.config

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    is_four_class = config.get("healthy_class_enabled", False) and not args.force_three_class

    if is_four_class:
        prepare_four_class_dataset(config, project_root, method=args.method)
    else:
        prepare_three_class_legacy(config, project_root, method=args.method)

if __name__ == "__main__":
    main()
