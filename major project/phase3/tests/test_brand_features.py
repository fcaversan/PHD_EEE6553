"""
Unit tests for Phase 3 brand-aware feature extraction.
"""

from src.brand_features import (
    brand_in_domain,
    brand_count,
    trust_word_in_domain,
    min_brand_edit_distance,
    get_hostname,
    get_registered_domain,
)


def test_hostname_and_registered_domain_extraction():
    hostname = get_hostname("https://accounts.google.com/signin/v2")
    assert hostname == "accounts.google.com"
    assert get_registered_domain(hostname) == "google.com"


def test_brand_in_domain_flags_impersonation():
    assert brand_in_domain("https://paypal-security-center.com/verify") == 1.0
    assert brand_in_domain("https://secure-login-google.com/auth") == 1.0


def test_brand_in_domain_not_flagging_legitimate():
    assert brand_in_domain("https://paypal.com/myaccount") == 0.0
    assert brand_in_domain("https://accounts.google.com/signin") == 0.0


def test_brand_count_detects_multiple_brands():
    assert brand_count("https://microsoft-office-github-login.com") >= 2.0
    assert brand_count("https://randomsite-example.org") == 0.0


def test_trust_word_in_domain_only():
    assert trust_word_in_domain("https://secure-paypal-login.com/path") == 1.0
    assert trust_word_in_domain("https://paypal.com/login") == 0.0


def test_min_brand_edit_distance_typosquatting():
    legit = min_brand_edit_distance("https://paypal.com")
    typo = min_brand_edit_distance("https://paypa1.com")
    unrelated = min_brand_edit_distance("https://randomdomainexample.xyz")

    assert legit == 0.0
    assert typo < unrelated
