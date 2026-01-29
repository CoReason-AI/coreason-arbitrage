from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from coreason_arbitrage.server import (
    app,
    MockBudgetClient,
    MockAuditClient,
    MockFoundryClient
)
from coreason_identity.models import UserContext


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


def test_chat_completions_permission_error(mock_acompletion: AsyncMock) -> None:
    # Simulate PermissionError (Budget Exceeded)
    with patch("coreason_arbitrage.smart_client.CompletionsWrapperAsync.create", side_effect=PermissionError("Budget exceeded")):
        with TestClient(app) as client:
            payload = {"messages": [{"role": "user", "content": "Hello"}], "user": "test_user"}
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 429
            assert response.json()["detail"] == "Budget exceeded"


def test_chat_completions_runtime_error(mock_acompletion: AsyncMock) -> None:
    # Simulate RuntimeError (Routing Failed)
    with patch("coreason_arbitrage.smart_client.CompletionsWrapperAsync.create", side_effect=RuntimeError("No healthy models")):
        with TestClient(app) as client:
            payload = {"messages": [{"role": "user", "content": "Hello"}], "user": "test_user"}
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 503
            assert response.json()["detail"] == "No healthy models"


def test_chat_completions_generic_error(mock_acompletion: AsyncMock) -> None:
    # Simulate Generic Exception
    with patch("coreason_arbitrage.smart_client.CompletionsWrapperAsync.create", side_effect=ValueError("Unexpected error")):
        with TestClient(app) as client:
            payload = {"messages": [{"role": "user", "content": "Hello"}], "user": "test_user"}
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 500
            assert response.json()["detail"] == "Unexpected error"


def test_mock_clients() -> None:
    # Coverage for Mock Clients
    budget_client = MockBudgetClient()
    assert budget_client.check_allowance("user") is True
    # Mock UserContext
    user_context = UserContext(user_id="user", tenant_id="tenant", groups=[], permissions=[], email="test@example.com")
    assert budget_client.get_remaining_budget_percentage(user_context) == 1.0
    budget_client.deduct_funds("user", 1.0)  # Should verify 'pass' line

    audit_client = MockAuditClient()
    audit_client.log_transaction("user", "model", 10, 10, 0.1)

    foundry_client = MockFoundryClient()
    models = foundry_client.list_custom_models()
    assert len(models) > 0
    assert models[0].id == "azure/gpt-4o"
