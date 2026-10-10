"""Moroccan Darija text normalizer and Arabizi-to-Arabic phonetic transliteration engine."""

from __future__ import annotations

import re

# Moroccan Darija Stopwords
DARIJA_STOPWORDS: set[str] = {
    "ديال", "ديالو", "ديالها", "ديالهم", "ديالنا",
    "واش", "شنو", "علاش", "فاش", "وقتاش", "كفاش", "فين",
    "حيت", "لحقاش", "حتا", "حتى", "راه", "راها", "راهم",
    "غادي", "غادية", "غاديين", "باغي", "باغية", "باغيين",
    "كاين", "كاينة", "كاينين", "مكاينش", "ماكاينش",
    "هاد", "هادا", "هادي", "هادو",
    "بزاف", "شوية", "غير", "بلاتي", "دابا",
    "مع", "ف", "في", "على", "من", "إلى", "يلا", "إلا",
    "و", "او", "أو", "ولا", "ولكن", "لكن",
    "انا", "أنا", "نتا", "نتيا", "هو", "هي", "حنا", "نتوما", "هما",
    "لي", "اللي", "الذي", "التي",
}

# Arabic Unicode Ranges
ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

# Arabizi numeral and digraph phonetic mappings to Arabic
ARABIZI_DIGRAPHS: list[tuple[str, str]] = [
    ("ch", "ش"),
    ("sh", "ش"),
    ("kh", "خ"),
    ("gh", "غ"),
    ("ou", "و"),
    ("th", "ث"),
    ("dh", "ذ"),
]

ARABIZI_CHAR_MAP: dict[str, str] = {
    "3": "ع",
    "7": "ح",
    "9": "ق",
    "5": "خ",
    "8": "غ",
    "2": "ا",
    "a": "ا",
    "b": "ب",
    "t": "ت",
    "j": "ج",
    "d": "د",
    "r": "ر",
    "z": "ز",
    "s": "س",
    "f": "ف",
    "q": "ق",
    "k": "ك",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "h": "ه",
    "w": "و",
    "y": "ي",
    "i": "ي",
    "u": "و",
    "o": "و",
    "e": "",  # Short vowels in Arabizi typically silent/omitted in consonantal Arabic
}


class DarijaNormalizer:
    """Preprocesses and standardizes Moroccan Darija written in Arabic or Arabizi."""

    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords

    @staticmethod
    def is_arabic_script(text: str) -> bool:
        """Return True if text primarily contains Arabic Unicode characters."""
        arabic_chars = len(ARABIC_CHAR_PATTERN.findall(text))
        total_chars = max(1, len(text.strip()))
        return (arabic_chars / total_chars) > 0.35

    @staticmethod
    def is_arabizi(text: str) -> bool:
        """Return True if text is written in Latin-script Arabizi with characteristic digits."""
        has_latin = bool(re.search(r"[a-zA-Z]", text))
        has_arabizi_digits = bool(re.search(r"[235789]", text))
        return has_latin and (has_arabizi_digits or not DarijaNormalizer.is_arabic_script(text))

    @staticmethod
    def squash_repeated_chars(text: str, max_repeat: int = 2) -> str:
        """Compress elongated characters (e.g. 'زوووووين' -> 'زوين', 'waaa3er' -> 'wa3er')."""
        return re.sub(r"(.)\1{" + str(max_repeat) + r",}", r"\1" * max_repeat, text)

    @classmethod
    def transliterate_arabizi_to_arabic(cls, text: str) -> str:
        """Phonetically map Arabizi words into Arabic script."""
        cleaned = text.lower()
        cleaned = cls.squash_repeated_chars(cleaned, max_repeat=2)

        words = cleaned.split()
        transliterated_words = []

        for word in words:
            # Handle common fixed Darija Arabizi words directly
            token = word
            # Step 1: Replace digraphs
            for digraph, replacement in ARABIZI_DIGRAPHS:
                token = token.replace(digraph, replacement)

            # Step 2: Replace individual Arabizi characters
            res_chars = []
            for ch in token:
                if ch in ARABIZI_CHAR_MAP:
                    res_chars.append(ARABIZI_CHAR_MAP[ch])
                else:
                    res_chars.append(ch)

            mapped_word = "".join(res_chars)
            transliterated_words.append(mapped_word)

        return " ".join(transliterated_words)

    @classmethod
    def normalize_arabic(cls, text: str) -> str:
        """Standardize Arabic script orthography and strip diacritics."""
        # 1. Remove URLs and mentions
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)

        # 2. Strip Tatweel (kashida)
        text = re.sub(r"ـ+", "", text)

        # 3. Strip Tashkeel / Harakat
        text = re.sub(r"[\u064B-\u0652\u0670]", "", text)

        # 4. Normalize Alef variants
        text = re.sub(r"[أإآٱ]", "ا", text)

        # 5. Normalize Taa Marbuta
        text = re.sub(r"ة\b", "ه", text)

        # 6. Normalize Alef Maqsura to Yaa
        text = re.sub(r"ى\b", "ي", text)

        # 7. Squash repeated letters
        text = cls.squash_repeated_chars(text, max_repeat=2)

        # 8. Strip non-Arabic symbols except basic whitespace
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

        # 9. Clean multi-spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize(self, text: str) -> str:
        """Execute full normalization pipeline on raw Darija text."""
        if not text or not text.strip():
            return ""

        # Check if text is Arabizi
        if self.is_arabizi(text):
            transliterated = self.transliterate_arabizi_to_arabic(text)
            normalized = self.normalize_arabic(transliterated)
        else:
            normalized = self.normalize_arabic(text)

        # Optional stopword removal
        if self.remove_stopwords:
            tokens = [t for t in normalized.split() if t not in DARIJA_STOPWORDS]
            normalized = " ".join(tokens)

        return normalized
