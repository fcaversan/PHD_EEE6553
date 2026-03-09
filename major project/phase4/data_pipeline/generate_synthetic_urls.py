"""
Synthetic Impersonation URL Generator
Generates realistic fake phishing URLs that impersonate known brands
using common attacker techniques (typosquatting, subdomain abuse,
path injection, homoglyph substitution, hyphenation).

These URLs are labeled as 'phishing' and provide the model with
direct supervision on brand-impersonation patterns — filling the
critical gap in the Kaggle dataset.

Usage:
    python generate_synthetic_urls.py --count 20000
    python generate_synthetic_urls.py --count 20000 --seed 42
"""

import os
import csv
import random
import string
import argparse
from typing import List, Tuple


# ── Brand list (reused from phase3/brand_features.py) ──────────────────

BRAND_TO_OFFICIAL_DOMAIN = {
    'google': 'google.com',
    'paypal': 'paypal.com',
    'apple': 'apple.com',
    'microsoft': 'microsoft.com',
    'amazon': 'amazon.com',
    'netflix': 'netflix.com',
    'facebook': 'facebook.com',
    'instagram': 'instagram.com',
    'twitter': 'twitter.com',
    'linkedin': 'linkedin.com',
    'dropbox': 'dropbox.com',
    'adobe': 'adobe.com',
    'github': 'github.com',
    'yahoo': 'yahoo.com',
    'outlook': 'outlook.com',
    'office': 'office.com',
    'chase': 'chase.com',
    'wellsfargo': 'wellsfargo.com',
    'citibank': 'citibank.com',
    'bankofamerica': 'bankofamerica.com',
    'hsbc': 'hsbc.com',
    'barclays': 'barclays.com',
    'natwest': 'natwest.com',
    'steam': 'steampowered.com',
    'spotify': 'spotify.com',
    'discord': 'discord.com',
    'twitch': 'twitch.tv',
    'ebay': 'ebay.com',
    'alibaba': 'alibaba.com',
    'walmart': 'walmart.com',
    'target': 'target.com',
    'fedex': 'fedex.com',
    'ups': 'ups.com',
    'dhl': 'dhl.com',
    'usps': 'usps.com',
    'irs': 'irs.gov',
    'nhs': 'nhs.uk',
    'vodafone': 'vodafone.com',
    'att': 'att.com',
    'verizon': 'verizon.com',
    'samsung': 'samsung.com',
    'huawei': 'huawei.com',
    'nvidia': 'nvidia.com',
    'intel': 'intel.com',
    'cisco': 'cisco.com',
    'oracle': 'oracle.com',
    'salesforce': 'salesforce.com',
    'zoom': 'zoom.us',
    'docusign': 'docusign.com',
    'coinbase': 'coinbase.com',
    'binance': 'binance.com',
    'blockchain': 'blockchain.com',
    'metamask': 'metamask.io',
}

TRUST_WORDS = [
    'secure', 'verify', 'login', 'account', 'update', 'auth', 'signin',
    'confirm', 'recover', 'support', 'password', 'billing', 'payment',
    'validate', 'credential', 'alert', 'urgent',
]

SUSPICIOUS_TLDS = [
    '.xyz', '.top', '.buzz', '.club', '.icu', '.site', '.online',
    '.info', '.tk', '.ml', '.ga', '.cf', '.gq', '.work', '.link',
    '.click', '.pw', '.bid', '.win', '.stream', '.loan', '.racing',
]

PHISHING_PATHS = [
    '/login', '/signin', '/verify', '/account', '/update',
    '/secure', '/auth', '/confirm', '/recover', '/password',
    '/billing', '/payment', '/validate', '/alert', '/webapps',
    '/myaccount', '/customer', '/identity', '/security',
    '/reset-password', '/unlock', '/suspension', '/reactivate',
]

# Common homoglyph substitutions attackers use
HOMOGLYPHS = {
    'a': ['@', '4', 'а'],   # 'а' is Cyrillic а (U+0430)
    'e': ['3', 'é', 'ё'],
    'i': ['1', 'l', '!'],
    'o': ['0', 'ö'],
    'l': ['1', 'I'],
    's': ['5', '$'],
    'g': ['9', 'q'],
    't': ['7'],
}


# ── Generation strategies ───────────────────────────────────────────────

def _rand_hex(length: int = 8) -> str:
    """Random hex string."""
    return ''.join(random.choices('0123456789abcdef', k=length))


def _rand_word(min_len: int = 4, max_len: int = 10) -> str:
    """Random lowercase alpha string."""
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def _rand_tld() -> str:
    """Pick a random suspicious TLD."""
    return random.choice(SUSPICIOUS_TLDS)


def _rand_path() -> str:
    """Pick a random phishing path, optionally with subpaths."""
    base = random.choice(PHISHING_PATHS)
    if random.random() < 0.4:
        base += '/' + _rand_word(3, 8)
    if random.random() < 0.3:
        base += '?id=' + _rand_hex(12)
    return base


def strategy_subdomain_abuse(brand: str) -> str:
    """
    Brand appears as subdomain of an unrelated domain.
    e.g., paypal.com.secure-login.xyz/verify
    """
    official = BRAND_TO_OFFICIAL_DOMAIN[brand]
    variants = [
        f"http://{official}.{_rand_word()}{_rand_tld()}{_rand_path()}",
        f"http://{brand}.{_rand_word()}{_rand_tld()}{_rand_path()}",
        f"http://{brand}-{random.choice(TRUST_WORDS)}.{_rand_word()}{_rand_tld()}{_rand_path()}",
        f"http://{brand}.{random.choice(TRUST_WORDS)}.{_rand_word()}{_rand_tld()}{_rand_path()}",
    ]
    return random.choice(variants)


def strategy_typosquat(brand: str) -> str:
    """
    Misspelled brand in domain.
    e.g., paypa1.com, paypall.com, papyal.com
    """
    name = brand
    mutation = random.choice(['swap', 'double', 'drop', 'insert', 'replace'])

    if mutation == 'swap' and len(name) > 2:
        # Swap two adjacent characters
        i = random.randint(0, len(name) - 2)
        name = name[:i] + name[i + 1] + name[i] + name[i + 2:]
    elif mutation == 'double' and len(name) > 1:
        # Double a character
        i = random.randint(0, len(name) - 1)
        name = name[:i] + name[i] * 2 + name[i + 1:]
    elif mutation == 'drop' and len(name) > 2:
        # Drop a character
        i = random.randint(1, len(name) - 1)
        name = name[:i] + name[i + 1:]
    elif mutation == 'insert':
        # Insert a random char
        i = random.randint(1, len(name) - 1)
        name = name[:i] + random.choice(string.ascii_lowercase) + name[i:]
    elif mutation == 'replace' and len(name) > 1:
        # Replace a character with an adjacent key or random
        i = random.randint(0, len(name) - 1)
        name = name[:i] + random.choice(string.ascii_lowercase) + name[i + 1:]

    # Use a suspicious or normal-looking TLD
    tld = random.choice([_rand_tld(), '.com', '.net', '.org'])
    return f"http://{name}{tld}{_rand_path()}"


def strategy_homoglyph(brand: str) -> str:
    """
    Replace characters with visually similar ones.
    e.g., g00gle.com, paypa1.com
    """
    name = list(brand)
    # Replace 1-3 characters that have homoglyphs
    replaceable = [(i, c) for i, c in enumerate(name) if c in HOMOGLYPHS]
    if replaceable:
        count = min(len(replaceable), random.randint(1, 2))
        targets = random.sample(replaceable, count)
        for i, c in targets:
            name[i] = random.choice(HOMOGLYPHS[c])

    result = ''.join(name)
    tld = random.choice(['.com', '.net', _rand_tld()])
    return f"http://{result}{tld}{_rand_path()}"


def strategy_hyphenation(brand: str) -> str:
    """
    Insert hyphens/trust words around brand.
    e.g., paypal-secure.com, secure-paypal-login.xyz
    """
    trust = random.choice(TRUST_WORDS)
    patterns = [
        f"{brand}-{trust}",
        f"{trust}-{brand}",
        f"{brand}-{trust}-{_rand_word(3, 6)}",
        f"{trust}-{brand}-{_rand_word(3, 6)}",
        f"{brand}{trust}",
        f"{trust}{brand}",
    ]
    domain = random.choice(patterns)
    tld = random.choice([_rand_tld(), '.com', '.net'])
    return f"http://{domain}{tld}{_rand_path()}"


def strategy_path_injection(brand: str) -> str:
    """
    Brand appears only in the URL path, not the domain.
    e.g., evil-site.xyz/paypal/login
    """
    domain = f"{_rand_word()}{_rand_tld()}"
    trust = random.choice(TRUST_WORDS)
    paths = [
        f"/{brand}/{trust}",
        f"/{brand}.com/{trust}",
        f"/{trust}/{brand}",
        f"/{brand}/{_rand_word()}/{trust}",
        f"/{brand}-{trust}",
    ]
    path = random.choice(paths)
    if random.random() < 0.3:
        path += '?token=' + _rand_hex(16)
    return f"http://{domain}{path}"


def strategy_ip_based(brand: str) -> str:
    """
    IP address with brand in path.
    e.g., http://192.168.1.1/paypal/login
    """
    ip = f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    trust = random.choice(TRUST_WORDS)
    paths = [
        f"/{brand}/{trust}",
        f"/{brand}.com/{trust}",
        f"/{brand}",
    ]
    port = random.choice(['', f':{random.randint(8000, 9999)}'])
    return f"http://{ip}{port}{random.choice(paths)}"


def strategy_long_subdomain(brand: str) -> str:
    """
    Very long subdomain chain to hide the actual domain.
    e.g., paypal.com.secure.verify.account.evil.xyz
    """
    chain_len = random.randint(2, 5)
    parts = [brand + '.com']
    for _ in range(chain_len):
        parts.append(random.choice(TRUST_WORDS + [_rand_word(3, 7)]))
    parts.append(_rand_word(5, 10) + _rand_tld())
    domain = '.'.join(parts)
    return f"http://{domain}{_rand_path()}"


# ── Benign synthetic URLs (to balance the dataset) ──────────────────────

BENIGN_DOMAINS = [
    'wikipedia.org', 'stackoverflow.com', 'reddit.com', 'medium.com',
    'bbc.co.uk', 'reuters.com', 'nytimes.com', 'cnn.com',
    'arxiv.org', 'nature.com', 'ieee.org', 'acm.org',
    'python.org', 'rust-lang.org', 'golang.org', 'nodejs.org',
    'example.com', 'example.org', 'w3.org', 'ietf.org',
    'university.edu', 'mit.edu', 'stanford.edu', 'ox.ac.uk',
    'weather.com', 'maps.google.com', 'translate.google.com',
    'docs.python.org', 'developer.mozilla.org', 'learn.microsoft.com',
]

BENIGN_PATHS = [
    '/wiki/', '/article/', '/news/', '/blog/', '/about',
    '/docs/', '/help/', '/faq/', '/contact', '/privacy',
    '/terms', '/search?q=', '/category/', '/topic/',
    '/en/', '/2024/', '/2025/', '/latest/', '/stable/',
]


def generate_benign_url() -> str:
    """Generate a realistic-looking benign URL."""
    domain = random.choice(BENIGN_DOMAINS)
    path = random.choice(BENIGN_PATHS)
    if random.random() < 0.5:
        path += _rand_word(4, 12)
    if random.random() < 0.2:
        path += '/' + _rand_word(3, 8)
    scheme = random.choice(['http://', 'https://'])
    return f"{scheme}{domain}{path}"


# ── Main generator ──────────────────────────────────────────────────────

STRATEGIES = [
    (strategy_subdomain_abuse, 0.25),
    (strategy_typosquat, 0.20),
    (strategy_homoglyph, 0.10),
    (strategy_hyphenation, 0.20),
    (strategy_path_injection, 0.10),
    (strategy_ip_based, 0.05),
    (strategy_long_subdomain, 0.10),
]


def generate_synthetic_phishing(count: int, seed: int = 42) -> List[dict]:
    """
    Generate synthetic impersonation phishing URLs.

    Args:
        count: Number of phishing URLs to generate
        seed: Random seed for reproducibility

    Returns:
        List of dicts with keys: url, type, source, target_brand, strategy
    """
    random.seed(seed)

    brands = list(BRAND_TO_OFFICIAL_DOMAIN.keys())
    strategy_funcs = [s[0] for s in STRATEGIES]
    strategy_weights = [s[1] for s in STRATEGIES]

    records = []
    seen_urls = set()

    attempts = 0
    max_attempts = count * 3  # avoid infinite loop

    while len(records) < count and attempts < max_attempts:
        attempts += 1

        brand = random.choice(brands)
        strategy_fn = random.choices(strategy_funcs, weights=strategy_weights, k=1)[0]

        try:
            url = strategy_fn(brand)
        except Exception:
            continue

        # Deduplicate
        if url in seen_urls:
            continue
        seen_urls.add(url)

        records.append({
            'url': url,
            'type': 'phishing',
            'source': 'synthetic',
            'target_brand': brand,
            'strategy': strategy_fn.__name__.replace('strategy_', ''),
        })

    print(f"Generated {len(records)} synthetic phishing URLs "
          f"({attempts} attempts, {len(seen_urls)} unique)")

    return records


def generate_synthetic_benign(count: int, seed: int = 42) -> List[dict]:
    """
    Generate synthetic benign URLs to maintain class balance.

    Args:
        count: Number of benign URLs to generate
        seed: Random seed

    Returns:
        List of dicts
    """
    random.seed(seed + 1000)  # Offset seed so it's different from phishing

    records = []
    seen = set()

    for _ in range(count * 2):
        if len(records) >= count:
            break
        url = generate_benign_url()
        if url not in seen:
            seen.add(url)
            records.append({
                'url': url,
                'type': 'benign',
                'source': 'synthetic',
                'target_brand': '',
                'strategy': 'benign_template',
            })

    print(f"Generated {len(records)} synthetic benign URLs")
    return records


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic impersonation phishing URLs"
    )
    parser.add_argument('--count', type=int, default=20000,
                        help='Number of phishing URLs to generate (default: 20000)')
    parser.add_argument('--benign-count', type=int, default=0,
                        help='Number of matching benign URLs (default: 0, i.e. phishing only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: processed/synthetic_phishing.csv)')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    output = args.output or os.path.join(processed_dir, "synthetic_phishing.csv")

    print(f"\n{'='*60}")
    print("SYNTHETIC URL GENERATOR")
    print(f"{'='*60}")
    print(f"  Phishing count: {args.count}")
    print(f"  Benign count:   {args.benign_count}")
    print(f"  Seed:           {args.seed}")
    print(f"  Brands:         {len(BRAND_TO_OFFICIAL_DOMAIN)}")
    print(f"  Strategies:     {len(STRATEGIES)}")

    # Generate
    all_records = generate_synthetic_phishing(args.count, args.seed)

    if args.benign_count > 0:
        all_records += generate_synthetic_benign(args.benign_count, args.seed)

    # Save
    fieldnames = ['url', 'type', 'source', 'target_brand', 'strategy']
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n  Output: {output}")

    # Strategy breakdown
    from collections import Counter
    strats = Counter(r['strategy'] for r in all_records if r['type'] == 'phishing')
    print(f"\n  Strategy distribution:")
    for s, c in strats.most_common():
        print(f"    {s:25s}: {c:6d} ({c/len(all_records)*100:.1f}%)")

    # Brand breakdown (top 10)
    brands = Counter(r['target_brand'] for r in all_records if r['target_brand'])
    print(f"\n  Top 10 targeted brands:")
    for b, c in brands.most_common(10):
        print(f"    {b:20s}: {c:6d}")

    # Show examples
    print(f"\n  Sample generated URLs:")
    for r in all_records[:10]:
        print(f"    [{r['strategy']:20s}] {r['url']}")

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
