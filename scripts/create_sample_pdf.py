#!/usr/bin/env python3
"""
Simple script to create a sample PDF from text file for testing ingestion.
"""

import os

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_pdf_from_text(text_file_path: str, pdf_file_path: str) -> None:
    """Convert a UTF-8 text file at ``text_file_path`` into a PDF at ``pdf_file_path``."""
    # Read the text file
    with open(text_file_path, encoding="utf-8") as f:
        content = f.read()

    # Create PDF
    c = canvas.Canvas(pdf_file_path, pagesize=letter)
    width, height = letter

    # Set up text formatting
    c.setFont("Helvetica", 12)

    # Split content into lines and write to PDF
    lines = content.split("\n")
    y_position = height - 50  # Start from top with margin

    for line in lines:
        if y_position < 50:  # Start new page if needed
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - 50

        c.drawString(50, y_position, line)
        y_position -= 20

    c.save()
    print(f"PDF created: {pdf_file_path}")


if __name__ == "__main__":
    text_file = "pdf_documents/sample_policy.txt"
    pdf_file = "pdf_documents/sample_policy.pdf"

    # Ensure directory exists
    os.makedirs(os.path.dirname(pdf_file), exist_ok=True)

    # Create a default sample text if missing
    if not os.path.exists(text_file):
        with open(text_file, "w", encoding="utf-8") as f:
            f.write("Sample Policy\n\nThis is an auto-generated sample policy document for tests.")

    create_pdf_from_text(text_file, pdf_file)
