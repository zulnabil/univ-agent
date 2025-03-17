import os
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable

from fastapi import UploadFile
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_milvus import BM25BuiltInFunction, Milvus

from app.config import settings
from app.core.embeddings import get_embeddings
from app.utils.logging import logger


class VectorStoreManager:
    _instance = None
    _vector_store = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
        return cls._instance

    @property
    def vector_store(self):
        if self._vector_store is None:
            logger.info(f"Initializing Milvus connection at {settings.MILVUS_URI}")
            embeddings = get_embeddings()
            self._vector_store = Milvus(
                embeddings,
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                consistency_level="Strong",
                connection_args={
                    "uri": settings.MILVUS_URI,
                    "token": settings.MILVUS_TOKEN,
                },
                collection_name=settings.MILVUS_COLLECTION,
                enable_dynamic_field=True,
                auto_id=True,
            )
        return self._vector_store


def get_vector_store():
    """Initialize and return the vector store."""
    return VectorStoreManager().vector_store


async def add_documents_to_vector_store(documents: list, hash: str, tag: str):
    """Add documents to the vector store."""
    try:
        logger.info(f"Adding {len(documents)} documents to vector store...")
        # check if hashes are already in the vector store
        hashes = [hash + f"_{i}" for i in range(len(documents))]
        vector_store = get_vector_store()
        existing_hashes = vector_store.get_pks(
            expr=f"pk in {hashes}",
        )
        if existing_hashes:
            raise ValueError("Hashes already in the vector store")

        # replace metadata with tag
        for doc in documents:
            doc.metadata = {"tag": tag}

        vector_store.auto_id = False
        await vector_store.aadd_documents(documents, ids=hashes)
        logger.info("Documents added successfully.")  # Log successful addition

    except Exception as e:
        logger.error(f"Error adding documents to vector store: {e}")
        raise e


@asynccontextmanager
async def temporary_file(content: bytes, suffix: str) -> AsyncGenerator[str, None]:
    """Context manager for handling temporary files from bytes content."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        temp_path = tmp_file.name

    try:
        yield temp_path
    finally:
        os.remove(temp_path)


async def process_file_with_loader(
    file: UploadFile, suffix: str, loader_class: Callable
) -> list:
    """Generic function to process files with a given loader."""
    try:
        async with temporary_file(file, suffix) as temp_path:
            loader = loader_class(temp_path)
            return loader.load()
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise e


async def get_vector_from_pdf(content: bytes):
    """Get the vector from the pdf file."""
    return await process_file_with_loader(content, ".pdf", PyPDFLoader)


async def get_vector_from_docx(content: bytes):
    """Get the vector from the docx file."""
    return await process_file_with_loader(content, ".docx", Docx2txtLoader)


async def get_vector_from_txt(content: bytes):
    """Get the vector from the txt file."""
    return await process_file_with_loader(content, ".txt", TextLoader)


async def get_vector_from_csv(content: bytes):
    """Get the vector from the csv file."""
    return await process_file_with_loader(content, ".csv", CSVLoader)
