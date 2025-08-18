"""set_turkish_language_default_and_update_existing

Revision ID: b5889e3a7cbb
Revises: 12ee97adc7c0
Create Date: 2025-08-18 12:24:15.496684

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b5889e3a7cbb"
down_revision: str | Sequence[str] | None = "12ee97adc7c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema to set Turkish as default language."""
    # Update database default for Turkish service
    op.alter_column("documents", "language", server_default="tr")

    # Update existing documents to correct language
    op.execute("UPDATE documents SET language = 'tr' WHERE language = 'en'")


def downgrade() -> None:
    """Downgrade schema to restore English default."""
    # Restore English default
    op.alter_column("documents", "language", server_default="en")

    # Restore existing documents to English (optional - may not be desired)
    op.execute("UPDATE documents SET language = 'en' WHERE language = 'tr'")
