from pathlib import Path
from typing import List

from llama_index import ServiceContext, StorageContext, VectorStoreIndex, download_loader
from llama_index.embeddings import OpenAIEmbedding
from llama_index.vector_stores import PGVectorStore

from app.core.config import settings


def get_pdf_files() -> List[Path]:
    """Get all PDF files from the documents directory."""
    pdf_dir = Path(settings.PDF_DOCUMENTS_DIR)
    if not pdf_dir.exists():
        raise Exception(f"PDF documents directory not found: {pdf_dir}")

    return list(pdf_dir.glob("*.pdf"))


def main():
    # Setup PostgreSQL vector store
    vector_store = PGVectorStore.from_params(
        database=settings.POSTGRES_DB,
        host=settings.POSTGRES_SERVER,
        password=settings.POSTGRES_PASSWORD,
        port=5432,
        user=settings.POSTGRES_USER,
        table_name="document_vectors",
        embed_dim=1536,  # OpenAI embedding dimension
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Setup embedding model
    embed_model = OpenAIEmbedding()
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    # Load PDF files
    PDFReader = download_loader("PDFReader")

    # Process each PDF file
    pdf_files = get_pdf_files()
    total_documents = 0

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        loader = PDFReader()
        documents = loader.load_data(file=pdf_file)

        # Create index for the current PDF
        VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            storage_context=storage_context,
        )

        total_documents += len(documents)
        print(f"Indexed {len(documents)} documents from {pdf_file}")

    print(f"\nTotal documents indexed: {total_documents}")


if __name__ == "__main__":
    main()
