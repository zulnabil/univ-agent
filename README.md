# University RAG API

A FastAPI-based Retrieval-Augmented Generation (RAG) system for university information, providing an OpenAI-compatible API interface.

## Features

- OpenAI-compatible chat completions API
- Streaming and non-streaming response support
- RAG system with tool-based information retrieval
- Milvus vector store integration
- DeepInfra LLM integration
- Colored logging system
- CORS support
- Health check endpoint

## Prerequisites

- Python 3.8+
- Milvus database
- DeepInfra API access

## Environment Variables

```env
# API Settings
API_KEY=your-secure-api-key

# LLM Settings
DEEPINFRA_API_TOKEN=your-deepinfra-token
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Milvus Settings
MILVUS_URI=your-milvus-uri
MILVUS_TOKEN=your-milvus-token
MILVUS_COLLECTION=univ_collections

# System Settings
DEBUG=False
LOG_LEVEL=INFO
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
uvicorn app.main:app --host localhost --port 8000 --reload
```

### API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions endpoint
- `GET /health` - Health check endpoint

### Chat Completion Request Format

```json
{
    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "messages": [
        {
            "role": "user",
            "content": "Your question here"
        }
    ],
    "stream": false
}
```

## Available Tools

The RAG system includes the following information retrieval tools:

1. `get_student_thesis` - Retrieve thesis information
2. `get_schedules` - Retrieve class schedules
3. `get_other_info` - Retrieve general university information

## Development

The project uses:
- FastAPI for the web framework
- LangChain for LLM integration
- LangGraph for RAG workflow
- Milvus for vector storage
- ColorLog for colored logging

## Error Handling

The API includes comprehensive error handling with:
- Detailed error messages
- Error type classification
- HTTP status codes
- Colored logging for different severity levels

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Add your license here]