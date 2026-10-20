"""
clean_dataset.py
----------------
Audits and detects exact and near-duplicate images in the rice nutrient dataset.
- Exact duplicates: Computed using SHA-256 byte-level hash.
- Near duplicates: Computed using perceptual hashing (ImageHash pHash) with orientation handling.
- Detects both within-class and cross-class duplicates.
- Generates review reports without deleting, moving, or altering any original files.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image
import imagehash
import pandas as pd

# Prevent DecompressionBombError on large images
Image.MAX_IMAGE_PIXELS = None

# Configurable perceptual hash distance threshold
# Default threshold: 6 (hamming distance <= 6 out of 64 bits indicates strong visual similarity)
DEFAULT_HASH_THRESHOLD = 6

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
    Rice leaf photos may be taken in portrait or landscape; canonicalizing
    orientation (height >= width) and evaluating 0/180 degree rotations ensures
    robust matching regardless of camera orientation.
    """
    with Image.open(filepath) as img:
        gray = img.convert("L")
        # Canonicalize to portrait orientation
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

def load_dataset_records(project_root, audit_csv_path, raw_dir):
    """Loads valid image records from audit CSV or falls back to directory walk."""
    records = []
    audit_file = Path(audit_csv_path)

    if audit_file.exists():
        df = pd.read_csv(audit_file)
        if "is_valid" in df.columns:
            df = df[df["is_valid"] == True]
        for _, row in df.iterrows():
            fpath = Path(row["file_path"])
            if fpath.exists():
                records.append({
                    "class_name": row["class_name"],
                    "file_name": row["file_name"],
                    "file_path": str(fpath.resolve()),
                    "file_size_bytes": int(row.get("file_size_bytes", fpath.stat().st_size))
                })
        print(f"Loaded {len(records)} valid records from {audit_file.resolve()}")
    else:
        print(f"Audit CSV not found at {audit_file}. Scanning {raw_dir} directly...")
        for root, _, files in os.walk(raw_dir):
            for file in files:
                fpath = Path(root) / file
                if fpath.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    parent_name = fpath.parent.name
                    class_name = "Nitrogen" if "nitrogen" in parent_name.lower() else (
                        "Phosphorus" if "phosphorus" in parent_name.lower() else (
                            "Potassium" if "potassium" in parent_name.lower() else parent_name
                        )
                    )
                    records.append({
                        "class_name": class_name,
                        "file_name": fpath.name,
                        "file_path": str(fpath.resolve()),
                        "file_size_bytes": fpath.stat().st_size
                    })
        print(f"Found {len(records)} images from directory scan.")

    return records

def find_exact_duplicates(records):
    """Groups files by exact SHA-256 hash."""
    sha_map = defaultdict(list)
    print("\n[1/3] Computing SHA-256 hashes for exact duplicate detection...")

    for idx, rec in enumerate(records, 1):
        if idx % 200 == 0 or idx == len(records):
            print(f"  Processed {idx}/{len(records)} files...")
        rec["sha256"] = compute_sha256(rec["file_path"])
        sha_map[rec["sha256"]].append(rec)

    exact_groups = []
    group_idx = 1

    for sha, group in sha_map.items():
        if len(group) > 1:
            classes = set(r["class_name"] for r in group)
            is_cross_class = len(classes) > 1
            group_id = f"EXACT_{group_idx:03d}"
            group_idx += 1
            for r in group:
                exact_groups.append({
                    "group_id": group_id,
                    "sha256": sha,
                    "is_cross_class": is_cross_class,
                    "class_name": r["class_name"],
                    "file_name": r["file_name"],
                    "file_path": r["file_path"],
                    "file_size_bytes": r["file_size_bytes"]
                })

    return exact_groups, sha_map

def find_near_duplicates(records, threshold=DEFAULT_HASH_THRESHOLD):
    """
    Computes perceptual hashes and finds near-duplicate pairs within distance threshold.
    Also groups near-duplicate components using disjoint set / union-find.
    """
    print(f"\n[2/3] Computing perceptual hashes (threshold <= {threshold})...")
    phash_dict = {}

    for idx, rec in enumerate(records, 1):
        if idx % 200 == 0 or idx == len(records):
            print(f"  Hashed {idx}/{len(records)} images...")
        phash_dict[rec["file_path"]] = compute_perceptual_hashes(rec["file_path"])

    print("\n[3/3] Performing pairwise perceptual hash comparisons...")
    n = len(records)
    total_comparisons = n * (n - 1) // 2
    print(f"  Total pairwise comparisons: {total_comparisons:,}")

    near_pairs = []
    # Parent pointer for union-find grouping
    parent = list(range(n))
    def find_p(i):
        if parent[i] == i:
            return i
        parent[i] = find_p(parent[i])
        return parent[i]
    def union_p(i, j):
        root_i = find_p(i)
        root_j = find_p(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        rec_a = records[i]
        ha = phash_dict[rec_a["file_path"]]
        for j in range(i + 1, n):
            rec_b = records[j]
            hb = phash_dict[rec_b["file_path"]]
            dist = min_phash_distance(ha, hb)

            if dist <= threshold:
                union_p(i, j)
                is_cross_class = rec_a["class_name"] != rec_b["class_name"]
                is_exact_sha = rec_a.get("sha256") == rec_b.get("sha256")

                if is_cross_class:
                    review_flag = "CONFLICTING_LABELS_CROSS_CLASS"
                elif is_exact_sha:
                    review_flag = "EXACT_DUPLICATE_SAME_CLASS"
                else:
                    review_flag = "NEAR_DUPLICATE_SAME_CLASS"

                near_pairs.append({
                    "idx_a": i,
                    "idx_b": j,
                    "file_a_name": rec_a["file_name"],
                    "file_a_class": rec_a["class_name"],
                    "file_a_path": rec_a["file_path"],
                    "file_b_name": rec_b["file_name"],
                    "file_b_class": rec_b["class_name"],
                    "file_b_path": rec_b["file_path"],
                    "hash_distance": dist,
                    "is_cross_class": is_cross_class,
                    "is_exact_sha256": is_exact_sha,
                    "review_flag": review_flag
                })

    # Group components
    cluster_map = defaultdict(list)
    for i in range(n):
        root = find_p(i)
        cluster_map[root].append(i)

    # Filter out singletons
    multi_clusters = {k: v for k, v in cluster_map.items() if len(v) > 1}
    cluster_id_lookup = {}
    for cluster_idx, (root, members) in enumerate(multi_clusters.items(), 1):
        c_id = f"NEAR_GRP_{cluster_idx:03d}"
        for member in members:
            cluster_id_lookup[member] = c_id

    for pair in near_pairs:
        pair["group_id"] = cluster_id_lookup.get(pair["idx_a"], "UNGROUPED")
        del pair["idx_a"]
        del pair["idx_b"]

    # Reorder columns
    ordered_pairs = []
    for p in near_pairs:
        ordered_pairs.append({
            "group_id": p["group_id"],
            "hash_distance": p["hash_distance"],
            "review_flag": p["review_flag"],
            "is_cross_class": p["is_cross_class"],
            "is_exact_sha256": p["is_exact_sha256"],
            "file_a_class": p["file_a_class"],
            "file_a_name": p["file_a_name"],
            "file_a_path": p["file_a_path"],
            "file_b_class": p["file_b_class"],
            "file_b_name": p["file_b_name"],
            "file_b_path": p["file_b_path"]
        })

    return ordered_pairs, len(multi_clusters)

def main():
    parser = argparse.ArgumentParser(description="Audit and detect duplicates in rice nutrient dataset.")
    parser.add_argument("--config", default="config.json", help="Path to configuration JSON")
    parser.add_argument("--threshold", type=int, default=DEFAULT_HASH_THRESHOLD,
                        help=f"Perceptual hash hamming distance threshold (default: {DEFAULT_HASH_THRESHOLD})")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    raw_dir = project_root / config.get("paths", {}).get("raw", "data/raw")
    outputs_dir = project_root / config.get("paths", {}).get("outputs", "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = outputs_dir / "dataset_audit.csv"

    print("=" * 75)
    print("RICE NUTRIENT AI - DUPLICATE DETECTION & REVIEW")
    print("=" * 75)
    print(f"Perceptual Hash Distance Threshold: {args.threshold}")
    print(f"Raw data root: {raw_dir}")
    print(f"Outputs dir  : {outputs_dir}")

    # Load records
    records = load_dataset_records(project_root, audit_csv, raw_dir)
    total_images = len(records)
    if total_images == 0:
        print("[ERROR] No image records found. Please run inspect_dataset.py first.")
        return

    # 1. Exact duplicates
    exact_duplicate_rows, sha_map = find_exact_duplicates(records)
    exact_df = pd.DataFrame(exact_duplicate_rows)
    exact_csv_path = outputs_dir / "exact_duplicates.csv"

    if not exact_df.empty:
        exact_df.to_csv(exact_csv_path, index=False)
        exact_groups_count = exact_df["group_id"].nunique()
        exact_dup_files_count = len(exact_df)
        exact_cross_class_groups = exact_df[exact_df["is_cross_class"] == True]["group_id"].nunique()
    else:
        # Create empty CSV with schema
        pd.DataFrame(columns=[
            "group_id", "sha256", "is_cross_class", "class_name",
            "file_name", "file_path", "file_size_bytes"
        ]).to_csv(exact_csv_path, index=False)
        exact_groups_count = 0
        exact_dup_files_count = 0
        exact_cross_class_groups = 0

    unique_exact_files = len(sha_map)

    # 2. Near duplicates
    near_duplicate_pairs, near_clusters_count = find_near_duplicates(records, threshold=args.threshold)
    near_df = pd.DataFrame(near_duplicate_pairs)
    near_csv_path = outputs_dir / "near_duplicate_review.csv"

    if not near_df.empty:
        near_df.to_csv(near_csv_path, index=False)
        near_pairs_count = len(near_df)
        near_cross_class_pairs = len(near_df[near_df["is_cross_class"] == True])
    else:
        pd.DataFrame(columns=[
            "group_id", "hash_distance", "review_flag", "is_cross_class",
            "is_exact_sha256", "file_a_class", "file_a_name", "file_a_path",
            "file_b_class", "file_b_name", "file_b_path"
        ]).to_csv(near_csv_path, index=False)
        near_pairs_count = 0
        near_cross_class_pairs = 0

    # 3. Summary Report
    print("\n" + "=" * 75)
    print("DUPLICATE AUDIT SUMMARY")
    print("=" * 75)
    print(f"Total images analyzed            : {total_images}")
    print(f"Unique files by SHA-256          : {unique_exact_files}")
    print(f"Exact duplicate groups           : {exact_groups_count}")
    print(f"Total files in exact groups      : {exact_dup_files_count}")
    print(f"Exact cross-class duplicate grps : {exact_cross_class_groups}")
    print("-" * 75)
    print(f"Near-duplicate candidate pairs   : {near_pairs_count}")
    print(f"Near-duplicate clusters/groups   : {near_clusters_count}")
    print(f"Near-duplicate cross-class pairs : {near_cross_class_pairs}")
    print("-" * 75)
    print(f"Exact duplicates report saved to : {exact_csv_path.resolve()}")
    print(f"Near-duplicates review saved to  : {near_csv_path.resolve()}")
    print("=" * 75)

    if exact_cross_class_groups > 0 or near_cross_class_pairs > 0:
        print("\n[ATTENTION] Cross-class duplicates were detected!")
        print("  These candidates have conflicting labels across classes.")
        print(f"  Please review '{near_csv_path.name}' and filter by is_cross_class == True.")

if __name__ == "__main__":
    main()
