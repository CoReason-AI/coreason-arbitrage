# Requirements

## Core Dependencies

These are the fundamental requirements for the `coreason-arbitrage` library:

- **python**: `>=3.12, <3.15`
- **litellm**: `^1.81.4` - For unified LLM provider interactions.
- **loguru**: `^0.7.2` - For structured logging.
- **pydantic**: `^2.12.5` - For data validation and settings management.
- **anyio**: `^4.12.1` - Asynchronous compatibility.
- **httpx**: `^0.28.1` - Async HTTP client.
- **aiofiles**: `^25.1.0` - File operations.
- **coreason-identity**: Internal identity management.

## Server Mode Dependencies

To run `coreason-arbitrage` as a standalone microservice, the following additional packages are required:

- **fastapi**: `^0.128.0` - High-performance web framework.
- **uvicorn**: `^0.40.0` - ASGI server implementation.
- **python-multipart**: `^0.0.22` - Multipart support for FastAPI.

## Development Dependencies

- **pytest**: `^9.0.2`
- **ruff**: `^0.14.14`
- **pre-commit**: `^4.5.1`
- **pytest-cov**: `^7.0.0`
- **mkdocs**: `^1.6.0`
- **mkdocs-material**: `^9.5.26`
- **mypy**: `^1.19.1`
- **types-aiofiles**: `^25.1.0`
- **pytest-asyncio**: `^1.3.0`
