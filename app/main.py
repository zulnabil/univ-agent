import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.api.endpoints import router as api_router
from app.config import settings
from app.utils.logging import logger

# Initialize FastAPI app
app = FastAPI(
    title="University RAG API",
    description="OpenAI-compatible API for University RAG System",
    version="1.0.0",
    root_path="/univ",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="University RAG API",
        version="1.0.0",
        description="OpenAI-compatible API for University RAG System",
        routes=app.routes,
    )

    # Add security scheme for API key
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {"type": "apiKey", "in": "query", "name": "api_key"}
    }

    # Apply security globally
    openapi_schema["security"] = [{"ApiKeyAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Include API routers with version prefix
app.include_router(api_router, prefix="/v1")


# Generic error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": type(exc).__name__,
                "param": None,
                "code": "internal_server_error",
            }
        },
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "University RAG API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": time.time(),
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting University RAG API")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down University RAG API")


# Run the app if executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="localhost", port=8000, reload=settings.DEBUG)
