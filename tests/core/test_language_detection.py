"""Tests for automatic language detection functionality."""

from app.core.language_detection import (
    LanguageDetector,
    detect_document_language,
)


class TestLanguageDetector:
    """Tests for the LanguageDetector class."""

    def test_detect_turkish_text(self):
        """Test detection of Turkish text."""
        detector = LanguageDetector(default_language="tr")

        turkish_text = """
        Bu belge Türkçe yazılmış bir metindir. İçerisinde çok sayıda Türkçe kelime
        ve karakter bulunmaktadır. Şirketimizin politikaları hakkında bilgi veren
        önemli bir dökümanDır. Çalışanların bu kuralları bilmeleri gereklidir.
        """

        result = detector.detect_language(turkish_text)
        assert result == "tr"

    def test_detect_english_text(self):
        """Test detection of English text."""
        detector = LanguageDetector(default_language="tr")

        english_text = """
        This document contains important information about company policies and
        procedures. All employees should read and understand these guidelines
        carefully. The management team has prepared this comprehensive guide
        to ensure workplace safety and efficiency.
        """

        result = detector.detect_language(english_text)
        assert result == "en"

    def test_short_text_returns_default(self):
        """Test that short texts return the default language."""
        detector = LanguageDetector(default_language="tr")

        short_text = "Hello"
        result = detector.detect_language(short_text)
        assert result == "tr"

    def test_empty_text_returns_default(self):
        """Test that empty text returns the default language."""
        detector = LanguageDetector(default_language="tr")

        result = detector.detect_language("")
        assert result == "tr"

    def test_mixed_language_with_turkish_dominance(self):
        """Test mixed language text with Turkish dominance."""
        detector = LanguageDetector(default_language="tr")

        mixed_text = """
        Bu belge hem Türkçe hem de English içerik barındırır. Ancak ağırlıklı
        olarak Türkçe yazılmıştır. Some English words are scattered throughout,
        ama genel olarak Türkçe karakterler ve kelimeler daha fazladır.
        """

        result = detector.detect_language(mixed_text)
        assert result == "tr"

    def test_filename_detection_turkish(self):
        """Test Turkish language detection from filename."""
        detector = LanguageDetector(default_language="en")

        filenames = [
            "document_tr.pdf",
            "rapor_turkish.docx",
            "belge_türkçe.txt",
            "policy_turkce.pdf",
        ]

        for filename in filenames:
            result = detector.detect_language_from_filename(filename)
            assert result == "tr", f"Failed for filename: {filename}"

    def test_filename_detection_english(self):
        """Test English language detection from filename."""
        detector = LanguageDetector(default_language="tr")

        filenames = ["document_en.pdf", "report_english.docx", "policy_eng.txt"]

        for filename in filenames:
            result = detector.detect_language_from_filename(filename)
            assert result == "en", f"Failed for filename: {filename}"

    def test_filename_no_language_indicator(self):
        """Test filename without language indicators."""
        detector = LanguageDetector(default_language="tr")

        filenames = ["document.pdf", "report.docx", "policy.txt", "12345.pdf"]

        for filename in filenames:
            result = detector.detect_language_from_filename(filename)
            assert result is None, f"Should return None for filename: {filename}"

    def test_turkish_characters_boost_score(self):
        """Test that Turkish characters increase Turkish score."""
        detector = LanguageDetector(default_language="en")

        # Text with Turkish characters
        text_with_turkish_chars = "Şirket çalışanları için önemli duyuru: İş güvenliği"
        # Same text without Turkish characters
        text_without_turkish_chars = "Sirket calisanlari icin onemli duyuru: Is guvenligi"

        score_with = detector._calculate_turkish_score(text_with_turkish_chars)
        score_without = detector._calculate_turkish_score(text_without_turkish_chars)

        assert score_with > score_without

    def test_confidence_threshold(self):
        """Test confidence threshold functionality."""
        detector = LanguageDetector(default_language="tr")

        # Ambiguous text that might not reach high confidence
        ambiguous_text = "Document report policy management system."

        # High threshold should return default
        result_high = detector.detect_language(ambiguous_text, confidence_threshold=0.9)
        assert result_high == "tr"

        # Lower threshold might detect English
        result_low = detector.detect_language(ambiguous_text, confidence_threshold=0.3)
        # This could be either 'en' or 'tr' depending on scoring, but shouldn't crash
        assert result_low in ["tr", "en"]


class TestDetectDocumentLanguage:
    """Tests for the convenience function."""

    def test_detect_with_filename_priority(self):
        """Test that filename detection takes priority."""
        # Turkish text but English filename indicator
        turkish_text = "Bu belge Türkçe yazılmış bir metindir."
        english_filename = "document_en.pdf"

        result = detect_document_language(turkish_text, english_filename, default="tr")
        assert result == "en"  # Filename should take priority

    def test_detect_with_content_fallback(self):
        """Test fallback to content analysis."""
        turkish_text = """
        Bu belge şirket politikaları hakkında önemli bilgiler içermektedir.
        Çalışanların bu kuralları dikkatli bir şekilde okumaları gerekir.
        """
        neutral_filename = "document.pdf"

        result = detect_document_language(turkish_text, neutral_filename, default="en")
        assert result == "tr"

    def test_detect_with_no_filename(self):
        """Test detection with no filename provided."""
        turkish_text = "Türkçe metin içeriği burada yer almaktadır."

        result = detect_document_language(turkish_text, "", default="en")
        assert result == "tr"

    def test_custom_default_language(self):
        """Test custom default language."""
        ambiguous_text = "test"  # Too short for reliable detection

        result = detect_document_language(ambiguous_text, "", default="fr")
        assert result == "fr"
