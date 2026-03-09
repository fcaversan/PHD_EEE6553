"""
Brand-impersonation feature utilities for Phase 3.
Adds explicit signals for brand misuse, trust-word bait, and typosquatting.
"""

from urllib.parse import urlparse


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

TRUST_WORDS = {
    'secure', 'verify', 'login', 'account', 'update', 'auth', 'signin',
    'confirm', 'recover', 'support', 'password', 'billing', 'payment',
    'validate', 'credential', 'alert', 'urgent',
}

MULTI_PART_SUFFIXES = {
    'co.uk', 'org.uk', 'ac.uk', 'gov.uk', 'com.au', 'net.au', 'org.au',
    'co.jp', 'com.br', 'com.mx',
}


def _normalize_url_for_parse(url: str) -> str:
    return url if url.startswith(('http://', 'https://')) else f'http://{url}'


def get_hostname(url: str) -> str:
    """Return lowercase hostname (without port), or empty string."""
    try:
        parsed = urlparse(_normalize_url_for_parse(url))
        return (parsed.hostname or '').lower()
    except Exception:
        return ''


def get_registered_domain(hostname: str) -> str:
    """
    Best-effort registered domain extraction without external dependencies.
    """
    if not hostname:
        return ''

    parts = [p for p in hostname.split('.') if p]
    if len(parts) < 2:
        return hostname

    suffix2 = '.'.join(parts[-2:])
    suffix3 = '.'.join(parts[-3:]) if len(parts) >= 3 else ''

    if suffix2 in MULTI_PART_SUFFIXES and len(parts) >= 3:
        return suffix3
    return suffix2


def get_registered_domain_name(hostname: str) -> str:
    """Return SLD-ish token used for brand distance checks."""
    registered = get_registered_domain(hostname)
    if not registered:
        return ''
    domain_parts = registered.split('.')
    return domain_parts[-2] if len(domain_parts) >= 2 else domain_parts[0]


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (0 if ca == cb else 1)
            curr_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = curr_row
    return prev_row[-1]


def brand_in_domain(url: str) -> float:
    """1.0 if hostname contains a known brand but registered domain is not official."""
    hostname = get_hostname(url)
    if not hostname:
        return 0.0

    registered_domain = get_registered_domain(hostname)
    for brand, official_domain in BRAND_TO_OFFICIAL_DOMAIN.items():
        if brand in hostname and registered_domain != official_domain:
            return 1.0
    return 0.0


def brand_count(url: str) -> float:
    """Count distinct known brand tokens appearing in full URL string."""
    url_lower = url.lower()
    found = {brand for brand in BRAND_TO_OFFICIAL_DOMAIN if brand in url_lower}
    return float(len(found))


def trust_word_in_domain(url: str) -> float:
    """1.0 if hostname contains any trust-word bait token."""
    hostname = get_hostname(url)
    if not hostname:
        return 0.0
    return float(1 if any(token in hostname for token in TRUST_WORDS) else 0)


def min_brand_edit_distance(url: str) -> float:
    """
    Normalized min Levenshtein distance between registered domain name and brand list.
    Returns value in [0, 1+] where lower is more brand-like.
    """
    domain_name = get_registered_domain_name(get_hostname(url))
    if not domain_name:
        return 1.0

    min_dist = min(_levenshtein_distance(domain_name, brand)
                   for brand in BRAND_TO_OFFICIAL_DOMAIN)
    norm = max(len(domain_name), 1)
    return float(min_dist / norm)


def extract_brand_features(url: str) -> dict:
    """Extract all Phase 3 brand-aware features for one URL."""
    return {
        'brand_in_domain': brand_in_domain(url),
        'brand_count': brand_count(url),
        'trust_word_in_domain': trust_word_in_domain(url),
        'min_brand_edit_distance': min_brand_edit_distance(url),
    }
