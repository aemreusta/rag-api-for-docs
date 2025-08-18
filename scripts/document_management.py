#!/usr/bin/env python3
"""
Document management utility script.

This script provides utilities for managing document processing,
including metadata extraction and retry functionality.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def extract_document_metadata(file_path: str) -> dict:
    """Extract metadata from a document file."""
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Import here to avoid issues if pypdf is not available
        from app.core.jobs import _extract_pdf_per_page

        if file_path.lower().endswith(".pdf"):
            text, page_texts, metadata = _extract_pdf_per_page(file_bytes)
        else:
            # Handle text files
            text = file_bytes.decode("utf-8", errors="ignore")
            word_count = len(text.split()) if text else 0
            metadata = {"page_count": 1, "word_count": word_count}
            page_texts = {1: text}

        # Detect language (local import to satisfy lint rules)
        from app.core.language_detection import detect_document_language

        detected_language = detect_document_language(
            text=text,
            filename=Path(file_path).name,
            default="tr",
        )
        metadata["detected_language"] = detected_language

        return {
            "file_path": file_path,
            "text_length": len(text),
            "page_count": metadata["page_count"],
            "word_count": metadata["word_count"],
            "detected_language": detected_language,
            "extracted_pages": len(page_texts),
        }

    except Exception as e:
        return {"file_path": file_path, "error": str(e), "error_type": type(e).__name__}


def test_language_detection(text: str, filename: str = "") -> dict:
    """Test language detection on provided text."""
    # Local import to avoid E402 when script adjusts sys.path
    from app.core.language_detection import detect_document_language

    detected_language = detect_document_language(text=text, filename=filename, default="tr")

    return {
        "text_sample": text[:100] + "..." if len(text) > 100 else text,
        "filename": filename,
        "detected_language": detected_language,
        "text_length": len(text),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Document management utility for enhanced processing"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract metadata command
    extract_parser = subparsers.add_parser("extract", help="Extract metadata from a document file")
    extract_parser.add_argument("file_path", help="Path to document file")

    # Language detection command
    lang_parser = subparsers.add_parser("detect-language", help="Test language detection on text")
    lang_parser.add_argument("text", help="Text to analyze")
    lang_parser.add_argument("--filename", help="Optional filename for context")

    # Demo command
    subparsers.add_parser("demo", help="Run demonstration of enhanced features")

    args = parser.parse_args()

    if args.command == "extract":
        print(f"Extracting metadata from: {args.file_path}")
        metadata = extract_document_metadata(args.file_path)

        if "error" in metadata:
            print(f"❌ Error: {metadata['error']} ({metadata['error_type']})")
            return 1

        print("✅ Metadata extraction successful:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    elif args.command == "detect-language":
        print("Testing language detection...")
        result = test_language_detection(args.text, args.filename or "")

        print("✅ Language detection results:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    elif args.command == "demo":
        print("🚀 Document Processing Enhancement Demo")
        print("=" * 50)

        # Test Turkish text detection
        print("\n1. Turkish Text Detection:")
        turkish_text = """
        Bu belge Türkçe yazılmış önemli bir dökümandir. Şirket politikaları
        hakkında bilgi vermektedir. Çalışanların dikkatli okuması gereklidir.
        """
        result = test_language_detection(turkish_text)
        print(f"   Detected Language: {result['detected_language']}")
        print(f"   Text Length: {result['text_length']} characters")

        # Test English text detection
        print("\n2. English Text Detection:")
        english_text = """
        This document contains important information about company policies.
        All employees should read and understand these guidelines carefully.
        """
        result = test_language_detection(english_text)
        print(f"   Detected Language: {result['detected_language']}")
        print(f"   Text Length: {result['text_length']} characters")

        # Test filename detection
        print("\n3. Filename-based Detection:")
        result = test_language_detection("Short text", "document_en.pdf")
        print(f"   Filename: {result['filename']}")
        print(f"   Detected Language: {result['detected_language']}")

        print("\n🎉 Demo completed successfully!")
        print("\nEnhanced Features Summary:")
        print("- ✅ Automatic page count extraction from PDFs")
        print("- ✅ Word count calculation during processing")
        print("- ✅ Language detection with Turkish/English support")
        print("- ✅ Filename-based language hints")
        print("- ✅ Retry mechanism for failed documents")
        print("- ✅ Comprehensive structured logging")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
