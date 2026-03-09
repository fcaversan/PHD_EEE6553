"""
Data Merger Pipeline
Combines the original Kaggle malicious_phish.csv with supplementary sources
(PhishTank and/or synthetic impersonation URLs) into a single training CSV.

The output CSV has columns:  url, type, source
where source ∈ {'kaggle', 'phishtank', 'synthetic'}

Deduplication is performed on the url column.

Usage:
    # Merge Kaggle + synthetic only (no PhishTank yet)
    python merge_datasets.py --kaggle-only

    # Merge Kaggle + PhishTank + synthetic
    python merge_datasets.py

    # Custom paths
    python merge_datasets.py --kaggle ../../datasets/malicious_phish.csv \
                             --phishtank processed/phishtank_phishing.csv \
                             --synthetic processed/synthetic_phishing.csv

    # Control synthetic count
    python merge_datasets.py --kaggle-only --synthetic-count 20000
"""

import os
import sys
import csv
import argparse
from collections import Counter


def load_kaggle_csv(path: str) -> list[dict]:
    """
    Load the original Kaggle malicious_phish.csv.
    Expected columns: url, type
    """
    records = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('url', '').strip()
            label = row.get('type', '').strip().lower()
            if url and label:
                records.append({
                    'url': url,
                    'type': label,
                    'source': 'kaggle',
                })
    print(f"  Kaggle: {len(records)} records loaded")
    return records


def load_supplementary_csv(path: str, source_name: str) -> list[dict]:
    """
    Load a supplementary CSV (PhishTank or synthetic).
    Expected columns: url, type, [source, target_brand, ...]
    """
    records = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('url', '').strip()
            label = row.get('type', '').strip().lower()
            if url and label:
                records.append({
                    'url': url,
                    'type': label,
                    'source': source_name,
                })
    print(f"  {source_name}: {len(records)} records loaded")
    return records


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove duplicate URLs, keeping the first occurrence."""
    seen = set()
    unique = []
    for r in records:
        url_key = r['url'].lower().strip().rstrip('/')
        if url_key not in seen:
            seen.add(url_key)
            unique.append(r)
    dropped = len(records) - len(unique)
    if dropped > 0:
        print(f"  Deduplication: removed {dropped} duplicates")
    return unique


def save_merged_csv(records: list[dict], output_path: str) -> None:
    """Save merged dataset as CSV with columns: url, type, source."""
    fieldnames = ['url', 'type', 'source']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                'url': r['url'],
                'type': r['type'],
                'source': r['source'],
            })
    print(f"  Saved merged dataset: {output_path}")


def print_summary(records: list[dict]) -> None:
    """Print a detailed summary of the merged dataset."""
    print(f"\n{'='*60}")
    print("MERGED DATASET SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Total records: {len(records)}")

    # By source
    sources = Counter(r['source'] for r in records)
    print(f"\n  By source:")
    for src, count in sources.most_common():
        pct = count / len(records) * 100
        print(f"    {src:15s}: {count:7d} ({pct:5.2f}%)")

    # By class
    classes = Counter(r['type'] for r in records)
    print(f"\n  By class:")
    for cls, count in sorted(classes.items()):
        pct = count / len(records) * 100
        print(f"    {cls:15s}: {count:7d} ({pct:5.2f}%)")

    # By class × source
    cross = Counter((r['type'], r['source']) for r in records)
    print(f"\n  Class × Source breakdown:")
    for (cls, src), count in sorted(cross.items()):
        print(f"    {cls:15s} [{src:10s}]: {count:7d}")


def main():
    parser = argparse.ArgumentParser(description="Merge URL datasets for Phase 4")
    parser.add_argument('--kaggle', type=str,
                        default='../../../datasets/malicious_phish.csv',
                        help='Path to Kaggle malicious_phish.csv')
    parser.add_argument('--external', type=str,
                        default='processed/external_phishing.csv',
                        help='Path to external phishing CSV (from download_phishing_urls.py)')
    parser.add_argument('--synthetic', type=str,
                        default='processed/synthetic_phishing.csv',
                        help='Path to synthetic phishing CSV')
    parser.add_argument('--output', type=str,
                        default='processed/merged_dataset.csv',
                        help='Output merged CSV path')
    parser.add_argument('--kaggle-only', action='store_true',
                        help='Skip external phishing data (use only Kaggle + synthetic)')
    parser.add_argument('--synthetic-count', type=int, default=20000,
                        help='Generate this many synthetic URLs if CSV not found')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for synthetic generation')

    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    kaggle_path = args.kaggle if os.path.isabs(args.kaggle) else os.path.join(script_dir, args.kaggle)
    external_path = args.external if os.path.isabs(args.external) else os.path.join(script_dir, args.external)
    synthetic_path = args.synthetic if os.path.isabs(args.synthetic) else os.path.join(script_dir, args.synthetic)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)

    processed_dir = os.path.join(script_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("DATA MERGER PIPELINE")
    print(f"{'='*60}")

    all_records = []

    # 1. Load Kaggle baseline
    print(f"\n[1/3] Loading Kaggle baseline...")
    if not os.path.exists(kaggle_path):
        print(f"  ERROR: Kaggle dataset not found: {kaggle_path}")
        sys.exit(1)
    all_records.extend(load_kaggle_csv(kaggle_path))

    # 2. Load external phishing data (GitHub Phishing.Database + OpenPhish)
    if args.kaggle_only:
        print(f"\n[2/3] Skipping external phishing data (--kaggle-only)")
    else:
        print(f"\n[2/3] Loading external phishing data...")
        if os.path.exists(external_path):
            all_records.extend(load_supplementary_csv(external_path, 'phishing_db'))
        else:
            print(f"  External CSV not found: {external_path}")
            print(f"  Run download_phishing_urls.py first, or use --kaggle-only")
            print(f"  Continuing without external data...")

    # 3. Load or generate synthetic URLs
    print(f"\n[3/3] Loading synthetic impersonation URLs...")
    if os.path.exists(synthetic_path):
        all_records.extend(load_supplementary_csv(synthetic_path, 'synthetic'))
    else:
        print(f"  Synthetic CSV not found. Generating {args.synthetic_count} URLs...")
        # Import and run the generator
        from generate_synthetic_urls import generate_synthetic_phishing
        synthetic_records = generate_synthetic_phishing(args.synthetic_count, args.seed)
        # Convert to our format
        for r in synthetic_records:
            all_records.append({
                'url': r['url'],
                'type': r['type'],
                'source': 'synthetic',
            })
        # Also save for reproducibility
        import csv as csv_mod
        with open(synthetic_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv_mod.DictWriter(f, fieldnames=['url', 'type', 'source', 'target_brand', 'strategy'])
            writer.writeheader()
            writer.writerows(synthetic_records)
        print(f"  Saved synthetic URLs for reproducibility: {synthetic_path}")

    # 4. Deduplicate
    print(f"\n  Pre-dedup total: {len(all_records)}")
    all_records = deduplicate(all_records)

    # 5. Shuffle deterministically
    import random
    random.seed(args.seed)
    random.shuffle(all_records)

    # 6. Save
    save_merged_csv(all_records, output_path)

    # 7. Print summary
    print_summary(all_records)

    print(f"\n{'='*60}")
    print("MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Output file: {output_path}")
    print(f"  Next step:   python src/train.py")


if __name__ == '__main__':
    main()
