"""
PhishTank Data Downloader
Downloads the PhishTank verified-online phishing database and
converts it to a CSV with columns: url, type, source, target_brand.

PhishTank provides a free bulk download of verified phishing URLs.
Registration is required at https://phishtank.org to get an API key.

Usage:
    python download_phishtank.py --api-key YOUR_API_KEY
    python download_phishtank.py --from-file raw/phishtank.json
    python download_phishtank.py

The script can either download directly via the PhishTank API,
or process a manually-downloaded JSON/CSV file.
"""

import os
import sys
import json
import argparse

# ---------------------------------------------------------------------------
# PhishTank API download
# ---------------------------------------------------------------------------

PHISHTANK_API_URL = (
    "http://data.phishtank.com/data/{api_key}/online-valid.json"
)


def download_phishtank_api(api_key: str, output_path: str) -> str:
    """
    Download the current PhishTank verified-online feed via the API.

    Args:
        api_key: PhishTank developer API key
        output_path: Path to save the raw JSON file

    Returns:
        Path to saved JSON file
    """
    import urllib.request

    url = PHISHTANK_API_URL.format(api_key=api_key)
    headers = {
        "User-Agent": "phishtank/phd_research_project"
    }

    print(f"Downloading PhishTank feed from API...")
    print(f"  URL: {url[:60]}...")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    with open(output_path, "wb") as f:
        f.write(data)

    size_mb = len(data) / (1024 * 1024)
    print(f"  Downloaded {size_mb:.1f} MB -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Manual download fallback
# ---------------------------------------------------------------------------

MANUAL_DOWNLOAD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════╗
║  MANUAL PHISHTANK DOWNLOAD INSTRUCTIONS                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Go to https://phishtank.org/developer_info.php           ║
║  2. Register / Sign in (free)                                ║
║  3. Download "online-valid.json" or "online-valid.csv"       ║
║  4. Save to: data_pipeline/raw/phishtank.json                ║
║     (or .csv)                                                ║
║  5. Re-run:                                                  ║
║     python download_phishtank.py                              ║
║     (or --from-file raw/phishtank.json)                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_phishtank_json(json_path: str) -> list[dict]:
    """
    Parse PhishTank JSON feed into a list of dicts.

    Each entry yields:
        {url, type='phishing', source='phishtank',
         target_brand, phish_id, verification_time}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"  Loaded {len(entries)} PhishTank entries from JSON")

    records = []
    for entry in entries:
        url = entry.get("url", "").strip()
        if not url:
            continue

        # Extract target brand (PhishTank calls it "target")
        target = entry.get("target", "")
        # Sometimes nested: details -> target
        if not target and isinstance(entry.get("details"), list):
            for detail in entry["details"]:
                if detail.get("announcing_network"):
                    target = detail.get("announcing_network", "")
                    break

        records.append({
            "url": url,
            "type": "phishing",
            "source": "phishtank",
            "target_brand": target if target else "unknown",
            "phish_id": entry.get("phish_id", ""),
            "verification_time": entry.get("verification_time", ""),
        })

    return records


def parse_phishtank_csv(csv_path: str) -> list[dict]:
    """
    Parse PhishTank CSV feed (alternative format).
    The CSV typically has columns: phish_id, url, phish_detail_url,
    submission_time, verified, verification_time, online, target
    """
    import csv

    records = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", "").strip()
            if not url:
                continue

            records.append({
                "url": url,
                "type": "phishing",
                "source": "phishtank",
                "target_brand": row.get("target", "unknown"),
                "phish_id": row.get("phish_id", ""),
                "verification_time": row.get("verification_time", ""),
            })

    print(f"  Loaded {len(records)} PhishTank entries from CSV")
    return records


def save_records_csv(records: list[dict], output_path: str) -> None:
    """Save parsed records to a clean CSV."""
    import csv

    fieldnames = ["url", "type", "source", "target_brand", "phish_id",
                  "verification_time"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"  Saved {len(records)} records -> {output_path}")


def auto_detect_raw_file(raw_dir: str) -> str | None:
    """Return first existing likely PhishTank file in raw/, else None."""
    candidates = [
        "phishtank.json",
        "phishtank.csv",
        "online-valid.json",
        "online-valid.csv",
    ]
    for name in candidates:
        path = os.path.join(raw_dir, name)
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download PhishTank phishing URLs")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--api-key", type=str,
                       help="PhishTank API key for direct download")
    group.add_argument("--from-file", type=str,
                       help="Path to already-downloaded PhishTank JSON/CSV file")

    args = parser.parse_args()

    # Ensure we're running from the data_pipeline directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    raw_dir = os.path.join(script_dir, "raw")
    processed_dir = os.path.join(script_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    output_csv = os.path.join(processed_dir, "phishtank_phishing.csv")

    fpath = None

    if args.api_key:
        # Download from API
        raw_json = os.path.join(raw_dir, "phishtank.json")
        download_phishtank_api(args.api_key, raw_json)
        records = parse_phishtank_json(raw_json)
    else:
        if args.from_file:
            fpath = args.from_file
            if not os.path.isabs(fpath):
                fpath = os.path.join(script_dir, fpath)
        else:
            fpath = auto_detect_raw_file(raw_dir)
            if fpath:
                print(f"  Auto-detected raw file: {fpath}")

        if not fpath or not os.path.exists(fpath):
            print("\nERROR: No PhishTank input file found.")
            print(MANUAL_DOWNLOAD_INSTRUCTIONS)
            sys.exit(1)

        if fpath.endswith(".json"):
            records = parse_phishtank_json(fpath)
        elif fpath.endswith(".csv"):
            records = parse_phishtank_csv(fpath)
        else:
            print(f"ERROR: Unsupported file format: {fpath}")
            print("  Expected .json or .csv")
            sys.exit(1)

    if not records:
        print("ERROR: No records parsed. Check input file.")
        sys.exit(1)

    # Save processed output
    save_records_csv(records, output_csv)

    # Summary
    brands = {}
    for r in records:
        b = r["target_brand"]
        brands[b] = brands.get(b, 0) + 1

    print(f"\n{'='*60}")
    print("PhishTank Download Summary")
    print(f"{'='*60}")
    print(f"  Total phishing URLs: {len(records)}")
    print(f"  Unique target brands: {len(brands)}")
    print(f"\n  Top 15 targeted brands:")
    for brand, count in sorted(brands.items(), key=lambda x: -x[1])[:15]:
        print(f"    {brand:30s}: {count:6d}")
    print(f"\n  Output: {output_csv}")


if __name__ == "__main__":
    main()
