"""Unit tests for Moroccan Darija normalizer and Arabizi transliteration."""

from src.darijapulse.preprocessing.normalizer import DarijaNormalizer


def test_script_detection():
    # Arabic script
    assert DarijaNormalizer.is_arabic_script("هاد التيليفون واعر بزاف") is True
    assert DarijaNormalizer.is_arabizi("هاد التيليفون واعر بزاف") is False

    # Arabizi with numbers
    assert DarijaNormalizer.is_arabizi("tbarkellah 3lik akhay service zwin") is True
    assert DarijaNormalizer.is_arabic_script("tbarkellah 3lik akhay service zwin") is False

    # Arabizi without numbers
    assert DarijaNormalizer.is_arabizi("khedma n9iya w zwin bzaf") is True


def test_character_elongation_compression():
    # Arabic repeated characters
    assert DarijaNormalizer.squash_repeated_chars("زوووووين", max_repeat=2) == "زووين"
    assert DarijaNormalizer.squash_repeated_chars("زوووووين", max_repeat=1) == "زوين"

    # Arabizi repeated characters
    assert DarijaNormalizer.squash_repeated_chars("waaaaa3er", max_repeat=2) == "waa3er"
    assert DarijaNormalizer.squash_repeated_chars("toooop", max_repeat=2) == "toop"


def test_arabizi_transliteration():
    # Number 3 -> ع ('ayn)
    trans = DarijaNormalizer.transliterate_arabizi_to_arabic("3jebni")
    assert "ع" in trans

    # Number 7 -> ح (ha)
    trans = DarijaNormalizer.transliterate_arabizi_to_arabic("7aja")
    assert "ح" in trans

    # Digraph ch -> ش
    trans = DarijaNormalizer.transliterate_arabizi_to_arabic("chwiya")
    assert "ش" in trans


def test_arabic_orthography_normalization():
    norm = DarijaNormalizer()
    # Strip tatweel and harakat
    raw = "تـَـطـْـبـِـيـقْ أَمْثَلْ"
    cleaned = norm.normalize_arabic(raw)
    assert "ـ" not in cleaned
    assert "َ" not in cleaned
    assert "ْ" not in cleaned
    assert "تطبيق" in cleaned
    assert "امثل" in cleaned
