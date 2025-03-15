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
from pymilvus import MilvusClient

from app.config import settings
from app.core.embeddings import get_embeddings
from app.utils.logging import logger


def get_vector_store():
    """Initialize and return the vector store."""
    logger.info(f"Connecting to Milvus at {settings.MILVUS_URI}")

    embeddings = get_embeddings()

    vector_store = Milvus(
        embeddings,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        consistency_level="Strong",
        connection_args={"uri": settings.MILVUS_URI, "token": settings.MILVUS_TOKEN},
        collection_name=settings.MILVUS_COLLECTION,
        enable_dynamic_field=True,
        auto_id=True,
    )

    return vector_store


def get_vector_store_client():
    """Get the vector store client."""
    return MilvusClient(
        uri=settings.MILVUS_URI,
        token=settings.MILVUS_TOKEN,
    )


async def add_documents_to_vector_store(documents: list, hash: str):
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

        vector_store.auto_id = False
        await vector_store.aadd_documents(documents, ids=hashes)
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
