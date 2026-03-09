"""
Quick external phishing stress test for Phase 4 model.
Runs a curated list of realistic URLs and prints predictions.
"""

import sys
from typing import List, Dict

from classify_url import classify_url


def get_test_urls() -> List[Dict[str, str]]:
    return [
        {"url": "https://accounts.google.com/signin/v2/", "tag": "likely_benign"},
        {"url": "https://secure-login-google.com/auth/session", "tag": "likely_phishing"},
        {"url": "https://paypal.com/myaccount/summary", "tag": "likely_benign"},
        {"url": "https://paypal-security-center.com/verify/account", "tag": "likely_phishing"},
        {"url": "https://www.microsoft.com/en-us/security", "tag": "likely_benign"},
        {"url": "https://microsoftonline-authentication.com/login", "tag": "likely_phishing"},
        {"url": "https://github.com/login", "tag": "likely_benign"},
        {"url": "https://github-secure-auth.com/session/recover", "tag": "likely_phishing"},
        {"url": "https://appleid.apple.com/", "tag": "likely_benign"},
        {"url": "https://appleid-verify-now.com/icloud/recovery", "tag": "likely_phishing"},
        {"url": "https://www.dropbox.com/login", "tag": "likely_benign"},
        {"url": "https://dropbox-file-share-secure.com/open", "tag": "likely_phishing"},
        {"url": "https://www.netflix.com/login", "tag": "likely_benign"},
        {"url": "https://netflix-account-security-center.com/signin", "tag": "likely_phishing"},
        {"url": "https://portal.office.com/", "tag": "likely_benign"},
        {"url": "https://office365-credential-check.com/owa/auth", "tag": "likely_phishing"},
    ]


def main() -> None:
    rows = get_test_urls()
    print("=" * 120)
    print("PHASE 4 — EXTERNAL PHISHING STRESS TEST")
    print("=" * 120)
    print(f"{'#':<3} {'tag':<16} {'pred':<12} {'conf':<8} url")
    print("-" * 120)

    pred_counts = {}

    for i, item in enumerate(rows, start=1):
        result = classify_url(item["url"], verbose=False)
        pred = result["predicted_class"]
        conf = result["confidence"]
        pred_counts[pred] = pred_counts.get(pred, 0) + 1

        print(f"{i:<3} {item['tag']:<16} {pred:<12} {conf:<8.4f} {item['url']}")

    print("-" * 120)
    print("Prediction counts:")
    for cls_name in sorted(pred_counts.keys()):
        print(f"  {cls_name:<12}: {pred_counts[cls_name]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
