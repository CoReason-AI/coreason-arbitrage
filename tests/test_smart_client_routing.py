# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from unittest.mock import MagicMock, patch

import pytest

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import AuditClient, BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier


@pytest.fixture
def configured_engine() -> ArbitrageEngine:
    engine = ArbitrageEngine()
    # Reset singleton state for tests (hacky but necessary if tests run in same process)
    # engine._models = {} # Registry handles models
    engine.registry.clear()
    engine.load_balancer._failures.clear()
    engine.load_balancer._cooldown_until.clear()

    # Mock clients
    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    mock_budget.get_remaining_budget_percentage.return_value = 0.5

    mock_audit = MagicMock(spec=AuditClient)
    mock_foundry = MagicMock()

    engine.configure(mock_budget, mock_audit, mock_foundry)

    # Register a dummy model so routing works
    model = ModelDefinition(
        id="test-model",
        provider="test-provider",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
    )
    engine.registry.register_model(model)

    return engine


def test_smart_client_routing_flow(configured_engine: ArbitrageEngine) -> None:
    client = configured_engine.get_client()
    messages = [{"role": "user", "content": "hello"}]

    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        mock_response = MagicMock()
        mock_completion.return_value = mock_response

        client.chat.completions.create(messages=messages, user="test_user")

        # We can't check response["model"] easily because completion returns a mock object now,
        # unless we mock the return value's structure or check the call args.
        # But we can verify routing by checking which model ID was passed to completion.

        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "test-model"

        # Verify budget check was called
        assert configured_engine.budget_client is not None
        configured_engine.budget_client.check_allowance.assert_called_with("test_user")  # type: ignore


def test_smart_client_budget_denial(configured_engine: ArbitrageEngine) -> None:
    assert configured_engine.budget_client is not None
    configured_engine.budget_client.check_allowance.return_value = False  # type: ignore

    client = configured_engine.get_client()
    messages = [{"role": "user", "content": "hello"}]

    with pytest.raises(PermissionError, match="Budget exceeded"):
        client.chat.completions.create(messages=messages, user="test_user")


def test_smart_client_budget_fail_closed(configured_engine: ArbitrageEngine) -> None:
    # Simulate DB error
    assert configured_engine.budget_client is not None
    configured_engine.budget_client.check_allowance.side_effect = Exception("DB Down")  # type: ignore

    client = configured_engine.get_client()
    messages = [{"role": "user", "content": "hello"}]

    with pytest.raises(PermissionError, match="Budget check failed"):
        client.chat.completions.create(messages=messages, user="test_user")


def test_smart_client_routing_complexity_high(
    configured_engine: ArbitrageEngine,
) -> None:
    # Register a high tier model
    high_model = ModelDefinition(
        id="high-model",
        provider="test-provider",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.1,
        cost_per_1k_output=0.2,
    )
    configured_engine.registry.register_model(high_model)

    client = configured_engine.get_client()
    # "Analyze" keyword triggers high complexity
    messages = [{"role": "user", "content": "Analyze this data."}]

    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        mock_completion.return_value = MagicMock()

        client.chat.completions.create(messages=messages, user="test_user")

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["model"] == "high-model"
