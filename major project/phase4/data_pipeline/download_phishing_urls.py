"""
Phishing URL Downloader — GitHub Phishing.Database + OpenPhish
Downloads verified phishing URLs from free, no-registration sources
and converts them to a CSV with columns: url, type, source.

Primary source:
  Phishing-Database/Phishing.Database (GitHub, MIT license)
  ~780K+ verified phishing links, updated every few hours.

Secondary source:
  OpenPhish community feed (free, ~500 URLs, 15-min updates)

Usage:
    python download_phishing_urls.py              # download both sources
    python download_phishing_urls.py --skip-openphish  # GitHub only
    python download_phishing_urls.py --from-file raw/phishing-links-ACTIVE.txt
"""

import os
import sys
import csv
import argparse
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Source URLs (no API key, no registration)
# ---------------------------------------------------------------------------

GITHUB_PHISHING_DB_URL = (
    "https://raw.githubusercontent.com/Phishing-Database/"
    "Phishing.Database/master/phishing-links-ACTIVE.txt"
)

OPENPHISH_COMMUNITY_FEED = "https://openphish.com/feed.txt"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_text_file(url: str, save_path: str, label: str) -> str | None:
    """
    Download a plain-text URL list. Returns path on success, None on failure.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (phd-research-project)"
    }

    print(f"  Downloading {label}...")
    print(f"    URL: {url[:80]}...")

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()

        with open(save_path, "wb") as f:
            f.write(data)

        size_mb = len(data) / (1024 * 1024)
        print(f"    Downloaded {size_mb:.1f} MB -> {save_path}")
        return save_path

    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"    WARN: Download failed: {e}")
        return None
    except Exception as e:
        print(f"    WARN: Unexpected error: {e}")
        return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_url_list(txt_path: str, source_name: str) -> list[dict]:
    """
    Parse a plain-text file with one URL per line.
    Returns list of dicts: {url, type, source}
    """
    records = []
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            url = line.strip()
            # Skip empty lines, comments, headers
            if not url or url.startswith("#") or url.startswith("//"):
                continue
            records.append({
                "url": url,
                "type": "phishing",
                "source": source_name,
            })

    print(f"    Parsed {len(records):,} phishing URLs from {source_name}")
    return records


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove duplicate URLs (case-insensitive, strip trailing /)."""
    seen = set()
    unique = []
    for r in records:
        key = r["url"].lower().strip().rstrip("/")
        if key not in seen:
            seen.add(key)
            unique.append(r)
    dropped = len(records) - len(unique)
    if dropped > 0:
        print(f"    Deduplication: removed {dropped:,} duplicates")
    return unique


def save_csv(records: list[dict], output_path: str) -> None:
    """Save records to CSV."""
    fieldnames = ["url", "type", "source"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"    Saved {len(records):,} records -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download phishing URLs (no registration required)"
    )
    parser.add_argument("--from-file", type=str, default=None,
                        help="Use a local file instead of downloading")
    parser.add_argument("--skip-openphish", action="store_true",
                        help="Skip OpenPhish community feed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    raw_dir = os.path.join(script_dir, "raw")
    processed_dir = os.path.join(script_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    output_csv = args.output or os.path.join(
        processed_dir, "external_phishing.csv"
    )

    print(f"\n{'='*60}")
    print("PHISHING URL DOWNLOADER")
    print(f"{'='*60}")

    all_records = []

    if args.from_file:
        # ── Local file mode ───────────────────────────────────────
        fpath = args.from_file
        if not os.path.isabs(fpath):
            fpath = os.path.join(script_dir, fpath)
        if not os.path.exists(fpath):
            print(f"\n  ERROR: File not found: {fpath}")
            sys.exit(1)
        all_records = parse_url_list(fpath, "phishing_database")

    else:
        # ── Download from GitHub Phishing.Database ────────────────
        print(f"\n[1] GitHub Phishing.Database (MIT license)")
        print(f"    ~780K+ verified active phishing links")
        raw_github = os.path.join(raw_dir, "phishing-links-ACTIVE.txt")
        result = download_text_file(
            GITHUB_PHISHING_DB_URL, raw_github, "Phishing.Database"
        )
        if result:
            all_records.extend(parse_url_list(raw_github, "phishing_database"))
        else:
            print("    Skipping (download failed)")

        # ── Download from OpenPhish community feed ────────────────
        if not args.skip_openphish:
            print(f"\n[2] OpenPhish community feed")
            print(f"    ~500 recently-detected phishing URLs")
            raw_openphish = os.path.join(raw_dir, "openphish-feed.txt")
            result = download_text_file(
                OPENPHISH_COMMUNITY_FEED, raw_openphish, "OpenPhish"
            )
            if result:
                all_records.extend(
                    parse_url_list(raw_openphish, "openphish")
                )
            else:
                print("    Skipping (download failed)")

    if not all_records:
        print("\nERROR: No phishing URLs collected.")
        sys.exit(1)

    # Deduplicate
    all_records = deduplicate(all_records)

    # Save
    save_csv(all_records, output_csv)

    # Summary
    from collections import Counter
    sources = Counter(r["source"] for r in all_records)

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Total phishing URLs: {len(all_records):,}")
    for src, count in sources.most_common():
        print(f"    {src:25s}: {count:,}")
    print(f"\n  Output: {output_csv}")
    print(f"\n  Next step:")
    print(f"    python merge_datasets.py")


if __name__ == "__main__":
    main()
