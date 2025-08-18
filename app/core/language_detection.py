"""
Automatic language detection for document processing.

This module provides language detection capabilities for documents,
with support for Turkish and other common languages.
"""

import re

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Common Turkish words for basic detection
TURKISH_WORDS = {
    "ve",
    "bir",
    "bu",
    "ile",
    "için",
    "olan",
    "den",
    "da",
    "de",
    "ı",
    "ın",
    "ün",
    "un",
    "lar",
    "ler",
    "dır",
    "dir",
    "dur",
    "dür",
    "ında",
    "ki",
    "gibi",
    "kadar",
    "sonra",
    "önce",
    "üzerinde",
    "altında",
    "arasında",
    "karşı",
    "şu",
    "o",
    "her",
    "tüm",
    "bütün",
    "çok",
    "az",
    "büyük",
    "küçük",
    "yeni",
    "eski",
    "iyi",
    "kötü",
    "doğru",
    "yanlış",
    "veya",
    "ama",
    "fakat",
    "çünkü",
    "eğer",
    "nasıl",
    "neden",
    "nerede",
    "ne",
    "kim",
    "hangi",
    "kaç",
    "ne zaman",
    "şey",
    "zaman",
    "yer",
    "kişi",
    "insan",
    "adam",
    "kadın",
    "çocuk",
    "anne",
    "baba",
    "aile",
    "ev",
    "iş",
    "okul",
    "ülke",
    "şehir",
    "dünya",
    "hayat",
    "para",
    "su",
    "yiyecek",
    "gün",
    "saat",
    "yıl",
    "ay",
    "hafta",
    "sabah",
    "akşam",
    "gece",
    "öğle",
    "başka",
    "diğer",
    "aynı",
    "farklı",
    "benzer",
    "önemli",
    "gerekli",
    "mümkün",
    "imkansız",
    "kolay",
    "zor",
    "hızlı",
    "yavaş",
    "uzun",
    "kısa",
    "geniş",
    "dar",
}

# Common English words for comparison
ENGLISH_WORDS = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "in",
    "is",
    "it",
    "you",
    "that",
    "he",
    "was",
    "for",
    "on",
    "are",
    "as",
    "with",
    "his",
    "they",
    "i",
    "at",
    "be",
    "this",
    "have",
    "from",
    "or",
    "one",
    "had",
    "by",
    "word",
    "but",
    "not",
    "what",
    "all",
    "were",
    "we",
    "when",
    "your",
    "can",
    "said",
    "there",
    "each",
    "which",
    "do",
    "how",
    "their",
    "if",
    "will",
    "up",
    "other",
    "about",
    "out",
    "many",
    "then",
    "them",
    "these",
    "so",
    "some",
    "her",
    "would",
    "make",
    "like",
    "into",
    "him",
    "has",
    "two",
    "more",
    "very",
    "after",
    "words",
    "first",
    "where",
    "much",
    "through",
    "back",
    "years",
    "work",
    "came",
    "right",
    "used",
    "take",
    "three",
    "states",
    "himself",
    "few",
    "house",
    "use",
    "during",
    "without",
    "again",
    "place",
    "american",
    "around",
    "however",
    "home",
    "small",
    "found",
}

# Turkish character patterns
TURKISH_CHARS = set("çğıöşüÇĞIİÖŞÜ")


class LanguageDetector:
    """Language detection service for documents."""

    def __init__(self, default_language: str = "tr"):
        """Initialize with default language."""
        self.default_language = default_language
        self.logger = get_logger(f"{__name__}.detector")

    def detect_language(self, text: str, confidence_threshold: float = 0.20) -> str:
        """
        Detect the language of given text.

        Args:
            text: Text to analyze
            confidence_threshold: Minimum confidence required for detection

        Returns:
            Language code ('tr', 'en', etc.) or default language if uncertain
        """
        if not text or len(text.strip()) < 30:
            self.logger.debug(
                "Text too short for language detection",
                text_length=len(text.strip()),
                default_language=self.default_language,
            )
            return self.default_language

        # Clean and normalize text
        cleaned_text = self._clean_text(text)

        # Calculate language scores
        turkish_score = self._calculate_turkish_score(cleaned_text)
        english_score = self._calculate_english_score(cleaned_text)

        self.logger.info(
            "Language detection scores calculated",
            turkish_score=round(turkish_score, 3),
            english_score=round(english_score, 3),
            text_length=len(cleaned_text),
            confidence_threshold=confidence_threshold,
        )

        # Determine language based on scores
        if turkish_score > english_score and turkish_score >= confidence_threshold:
            detected_language = "tr"
        elif english_score > turkish_score and english_score >= confidence_threshold:
            detected_language = "en"
        else:
            detected_language = self.default_language
            self.logger.info(
                "Language detection uncertain, using default",
                turkish_score=round(turkish_score, 3),
                english_score=round(english_score, 3),
                default_language=self.default_language,
            )

        self.logger.info(
            "Language detection completed",
            detected_language=detected_language,
            turkish_score=round(turkish_score, 3),
            english_score=round(english_score, 3),
            text_sample=text[:100] + "..." if len(text) > 100 else text,
        )

        return detected_language

    def _clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace and punctuation for word analysis
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _calculate_turkish_score(self, text: str) -> float:
        """Calculate Turkish language confidence score."""
        words = text.split()
        if not words:
            return 0.0

        # Score based on Turkish-specific characters (boosted weight)
        char_score = sum(1 for char in text if char in TURKISH_CHARS) / len(text)

        # Score based on common Turkish words (improved scoring)
        turkish_word_count = sum(1 for word in words if word in TURKISH_WORDS)
        word_score = turkish_word_count / len(words)

        # Turkish suffix patterns (enhanced detection)
        suffix_patterns = [
            r"\w+lar\b",
            r"\w+ler\b",  # plural suffixes
            r"\w+dır\b",
            r"\w+dir\b",
            r"\w+dur\b",
            r"\w+dür\b",  # copula
            r"\w+ında\b",
            r"\w+inde\b",
            r"\w+unda\b",
            r"\w+ünde\b",  # locative
            r"\w+dan\b",
            r"\w+den\b",
            r"\w+tan\b",
            r"\w+ten\b",  # ablative
            r"\w+nın\b",
            r"\w+nin\b",
            r"\w+nun\b",
            r"\w+nün\b",  # genitive
            r"\w+ı\b",
            r"\w+i\b",
            r"\w+u\b",
            r"\w+ü\b",  # accusative
        ]

        suffix_matches = 0
        for pattern in suffix_patterns:
            suffix_matches += len(re.findall(pattern, text))

        suffix_score = min(suffix_matches / len(words), 0.4)  # Increased cap

        # Boost score for Turkish characters more aggressively
        char_boost = min(char_score * 3.0, 0.6) if char_score > 0 else 0

        # Combined score with improved weights
        total_score = (char_boost * 0.4) + (word_score * 0.4) + (suffix_score * 0.2)

        return min(total_score, 1.0)

    def _calculate_english_score(self, text: str) -> float:
        """Calculate English language confidence score."""
        words = text.split()
        if not words:
            return 0.0

        # Score based on common English words (improved scoring)
        english_word_count = sum(1 for word in words if word in ENGLISH_WORDS)
        word_score = english_word_count / len(words)

        # Heavily penalize Turkish characters in English text
        turkish_char_penalty = sum(1 for char in text if char in TURKISH_CHARS) / len(text)

        # English patterns (enhanced detection)
        english_patterns = [
            r"\bthe\b",
            r"\band\b",
            r"\bof\b",
            r"\bto\b",
            r"\ba\b",
            r"\bin\b",
            r"\bis\b",
            r"\bit\b",
            r"\byou\b",
            r"\bthat\b",
            r"\bwith\b",
            r"\bfor\b",
            r"\bare\b",
            r"\bas\b",
            r"\bthis\b",
            r"\bhave\b",
            r"\bfrom\b",
            r"\bor\b",
            r"\bone\b",
            r"\bbut\b",
        ]

        pattern_matches = 0
        for pattern in english_patterns:
            pattern_matches += len(re.findall(pattern, text))

        pattern_score = min(pattern_matches / len(words), 0.4)  # Increased cap

        # English suffix patterns
        english_suffixes = [
            r"\w+ing\b",
            r"\w+tion\b",
            r"\w+ness\b",
            r"\w+able\b",
            r"\w+ible\b",
            r"\w+ous\b",
            r"\w+ful\b",
        ]

        suffix_matches = 0
        for pattern in english_suffixes:
            suffix_matches += len(re.findall(pattern, text))

        suffix_score = min(suffix_matches / len(words), 0.3)

        # Combined score with stronger penalty for Turkish characters
        total_score = (
            (word_score * 0.5)
            + (pattern_score * 0.3)
            + (suffix_score * 0.2)
            - (turkish_char_penalty * 1.0)
        )

        return max(total_score, 0.0)

    def detect_language_from_filename(self, filename: str) -> str | None:
        """
        Attempt to detect language from filename patterns.

        Args:
            filename: Document filename

        Returns:
            Language code if detectable from filename, None otherwise
        """
        filename_lower = filename.lower()

        # Look for language indicators in filename
        if any(
            indicator in filename_lower for indicator in ["_tr", "_turkish", "_turkce", "türkçe"]
        ):
            self.logger.info(
                "Language detected from filename", filename=filename, detected_language="tr"
            )
            return "tr"

        if any(indicator in filename_lower for indicator in ["_en", "_english", "_eng"]):
            self.logger.info(
                "Language detected from filename", filename=filename, detected_language="en"
            )
            return "en"

        return None


def detect_document_language(text: str, filename: str = "", default: str = "tr") -> str:
    """
    Convenience function for document language detection.

    Args:
        text: Document text content
        filename: Document filename (optional)
        default: Default language if detection is uncertain

    Returns:
        Detected language code
    """
    detector = LanguageDetector(default_language=default)

    # First try filename detection
    if filename:
        filename_lang = detector.detect_language_from_filename(filename)
        if filename_lang:
            return filename_lang

    # Fall back to content analysis
    return detector.detect_language(text)
