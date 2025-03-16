from typing import List

from fastapi import HTTPException, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm
from app.core.vector_store import (
    add_documents_to_vector_store,
    get_vector_from_csv,
    get_vector_from_docx,
    get_vector_from_pdf,
    get_vector_from_txt,
)
from app.utils.helpers import get_hash_from_bytes, validate_file_type
from app.utils.logging import logger
from app.utils.prompts import generate_tag_prompt


class DocumentService:
    def __init__(self):
        self.content_type_handlers = {
            "application/pdf": get_vector_from_pdf,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": get_vector_from_docx,
            "text/plain": get_vector_from_txt,
            "text/csv": get_vector_from_csv,
        }

    async def process_documents(self, files: List[UploadFile]) -> dict:
        """Upload multiple documents, vectorize them and store them in the vector store"""
        try:
            results = []
            for file in files:
                result = await self._process_single_file(file)
                results.append(result)

            return {
                "status": "completed",
                "total_files": len(files),
                "results": results,
            }
        except Exception as e:
            logger.error(f"Error in bulk upload: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _process_single_file(self, file: UploadFile) -> dict:
        """Process a single file and return its processing result."""
        try:
            validate_file_type(file)
            logger.info(f"Processing file: {file.filename}")
            logger.info(file.content_type)
            content = await file.read()
            hash = get_hash_from_bytes(content)
            docs = await self._get_vector_by_content_type(content, file.content_type)
            logger.info(f"Extracted {len(docs)} documents from {file.filename}")
            tag = await self._get_tag_by_document(docs)
            await add_documents_to_vector_store(docs, hash, tag)

            return {
                "filename": file.filename,
                "status": "success",
            }
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            return {"filename": file.filename, "status": "error", "error": str(e)}

    async def _get_vector_by_content_type(self, content: bytes, content_type: str):
        """Get vector based on file content type."""

        handler = self.content_type_handlers.get(content_type)
        if not handler:
            raise ValueError(f"Unsupported content type: {content_type}")

        return await handler(content)

    async def _get_tag_by_document(self, docs: list):
        """Get tag by document."""
        num_chunks = min(5, len(docs))
        content = "\n".join([doc.page_content for doc in docs[:num_chunks]])
        llm = get_llm()
        tag = await llm.ainvoke(
            [
                SystemMessage(generate_tag_prompt),
                HumanMessage(content=content),
            ]
        )
        return tag.content
