from pathlib import Path
from typing import List

import chromadb
from llama_index import ServiceContext, StorageContext, download_loader
from llama_index.embeddings import OpenAIEmbedding
from llama_index.vector_stores import ChromaVectorStore

from app.core.config import settings


def get_pdf_files() -> List[Path]:
    """Get all PDF files from the documents directory."""
    pdf_dir = Path(settings.PDF_DOCUMENTS_DIR)
    if not pdf_dir.exists():
        raise Exception(f"PDF documents directory not found: {pdf_dir}")

    return list(pdf_dir.glob("*.pdf"))


def main():
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("documents")

    # Setup vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Setup embedding model
    embed_model = OpenAIEmbedding()
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    # Load PDF files
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()

    # Process each PDF file
    pdf_files = get_pdf_files()
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        documents = loader.load_data(file=pdf_file)

        # Create index for the documents
        from llama_index import VectorStoreIndex

        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
        )

        # Save index for future use
        index.storage_context.persist(persist_dir="./storage")
        print(f"Indexed {len(documents)} documents from {pdf_file}")


if __name__ == "__main__":
    main()
