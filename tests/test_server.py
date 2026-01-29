from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from coreason_arbitrage.server import app


class Usage(BaseModel):
    prompt_tokens: int = 10
    completion_tokens: int = 10


class Response(BaseModel):
    usage: Usage = Usage()
    choices: List[Dict[str, Any]] = [{"message": {"content": "Hello"}}]


@pytest.fixture
def mock_acompletion() -> Any:
    with patch("coreason_arbitrage.smart_client.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = Response()
        yield mock


def test_health_check() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "routing_engine": "active"}


def test_chat_completions_success(mock_acompletion: AsyncMock) -> None:
    with TestClient(app) as client:
        payload = {"messages": [{"role": "user", "content": "Hello"}], "user": "test_user", "temperature": 0.5}
        response = client.post("/v1/chat/completions", json=payload)

        # Verify success
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello"

        # Verify SmartClient was called with correct arguments
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["messages"] == payload["messages"]
        # 'user' is passed in kwargs to acompletion in SmartClient
        assert call_kwargs["user"] == "test_user"
        assert call_kwargs["temperature"] == 0.5
