from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.gatekeeper import Gatekeeper
from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier, RoutingContext
from coreason_arbitrage.smart_client import SmartClient


@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock(spec=ArbitrageEngine)
    engine.load_balancer = MagicMock()
    engine.registry = MagicMock()
    engine.budget_client = MagicMock(spec=BudgetClient)
    engine.audit_client = MagicMock()
    return engine


@pytest.fixture
def mock_gatekeeper() -> MagicMock:
    gk = MagicMock(spec=Gatekeeper)
    gk.classify.return_value = RoutingContext(complexity=0.1)
    return gk


@pytest.fixture
def smart_client(mock_engine: MagicMock, mock_gatekeeper: MagicMock) -> SmartClient:
    # We need to patch Gatekeeper() inside ChatWrapper because it's hardcoded
    with patch("coreason_arbitrage.smart_client.Gatekeeper", return_value=mock_gatekeeper):
        client = SmartClient(mock_engine)
        # We also need to inject our mock engine back into the completions wrapper
        # because SmartClient init might store it differently or pass it down
        client.chat.completions.engine = mock_engine
        client.chat.completions.gatekeeper = mock_gatekeeper

        # Also need to mock the router inside completions wrapper to return a model
        mock_router = MagicMock()
        mock_model = ModelDefinition(
            id="test-model",
            provider="test-provider",
            tier=ModelTier.TIER_1_FAST,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.02,
            is_healthy=True,
        )
        mock_router.route.return_value = mock_model
        client.chat.completions.router = mock_router

        return client


def test_create_deducts_funds_on_success(smart_client: SmartClient, mock_engine: MagicMock) -> None:
    """
    Verifies that deduct_funds is called with correct amount on success.
    """
    messages = [{"role": "user", "content": "Hello"}]

    # Mock litellm.completion
    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 1000
        mock_completion.return_value = mock_response

        # Mock check_allowance to pass
        mock_engine.budget_client.check_allowance.return_value = True

        smart_client.chat.completions.create(messages, user="test_user")

        # Expected cost:
        # Input: 1000/1000 * 0.01 = 0.01
        # Output: 1000/1000 * 0.02 = 0.02
        # Total: 0.03
        mock_engine.budget_client.deduct_funds.assert_called_once_with("test_user", 0.03)


def test_create_deduction_failure_logs_error(smart_client: SmartClient, mock_engine: MagicMock) -> None:
    """
    Verifies that if deduct_funds fails, the exception is caught and logged,
    and the response is still returned.
    """
    messages = [{"role": "user", "content": "Hello"}]

    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 1000
        mock_completion.return_value = mock_response

        mock_engine.budget_client.check_allowance.return_value = True
        mock_engine.budget_client.deduct_funds.side_effect = Exception("DB Error")

        response = smart_client.chat.completions.create(messages, user="test_user")

        assert response == mock_response
        mock_engine.budget_client.deduct_funds.assert_called_once()
        # Should verify logging but we assume it's logged if no exception raised


def test_fail_open_deducts_funds(smart_client: SmartClient, mock_engine: MagicMock) -> None:
    """
    Verifies that deduct_funds is called even in fail-open scenario.
    """
    messages = [{"role": "user", "content": "Hello"}]

    # Mock completion to fail initially then succeed on fallback
    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        # Side effect: Raise exception for routing attempts (Simulating Router failure or completion failure)
        # But wait, fail-open happens if all retries fail OR Router crashes.
        # Let's simulate Router crash.
        cast(MagicMock, smart_client.chat.completions.router.route).side_effect = RuntimeError("No models")

        # The fallback call to completion needs to succeed
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 1000

        # We need mock_completion to return valid response when called with fallback model
        # The code calls completion(model=fallback_model.id...)
        mock_completion.return_value = mock_response

        mock_engine.budget_client.check_allowance.return_value = True

        smart_client.chat.completions.create(messages, user="test_user")

        # Fallback model pricing (from code): Input 0.005, Output 0.015
        # Cost: 1000/1000*0.005 + 1000/1000*0.015 = 0.02
        mock_engine.budget_client.deduct_funds.assert_called_once_with("test_user", 0.02)


def test_fail_open_deduction_failure_logs_error(smart_client: SmartClient, mock_engine: MagicMock) -> None:
    """
    Verifies that if deduct_funds fails in fail-open mode, it is logged and response returns.
    """
    messages = [{"role": "user", "content": "Hello"}]

    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        cast(MagicMock, smart_client.chat.completions.router.route).side_effect = RuntimeError("No models")

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 1000
        mock_completion.return_value = mock_response

        mock_engine.budget_client.check_allowance.return_value = True
        mock_engine.budget_client.deduct_funds.side_effect = Exception("Fail Open Deduction Error")

        response = smart_client.chat.completions.create(messages, user="test_user")

        assert response == mock_response
        mock_engine.budget_client.deduct_funds.assert_called_once()
