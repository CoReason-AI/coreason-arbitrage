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
    with patch("coreason_arbitrage.smart_client.Gatekeeper", return_value=mock_gatekeeper):
        client = SmartClient(mock_engine)
        client.chat.completions._async.engine = mock_engine
        client.chat.completions._async.gatekeeper = mock_gatekeeper

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


def test_audit_failure_does_not_block_deduction(smart_client: SmartClient, mock_engine: MagicMock) -> None:
    """
    Verify that if audit_client.log_transaction fails, budget_client.deduct_funds is still called.
    """
    messages = [{"role": "user", "content": "Hello"}]

    with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 1000
        mock_completion.return_value = mock_response

        mock_engine.budget_client.check_allowance.return_value = True

        # Simulate Audit Failure
        mock_engine.audit_client.log_transaction.side_effect = Exception("Audit Error")

        smart_client.chat.completions.create(messages, user="test_user")

        # Verify Audit was called
        mock_engine.audit_client.log_transaction.assert_called_once()

        # Verify Budget Deduction was STILL called
        # Cost: 1000/1000 * 0.01 + 1000/1000 * 0.02 = 0.03
        mock_engine.budget_client.deduct_funds.assert_called_once_with(user_id="test_user", amount=0.03)


def test_missing_usage_returns_response_skips_deduction(smart_client: SmartClient, mock_engine: MagicMock) -> None:
    """
    Verify that if response.usage is missing, the response is returned, and accounting is skipped safely.
    """
    messages = [{"role": "user", "content": "Hello"}]

    with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
        mock_response = MagicMock()
        # Delete usage attribute to simulate missing data
        del mock_response.usage

        mock_completion.return_value = mock_response
        mock_engine.budget_client.check_allowance.return_value = True

        response = smart_client.chat.completions.create(messages, user="test_user")

        # Should return the response successfully
        assert response == mock_response

        # Should NOT call deduct_funds (skipped due to error in cost calc)
        mock_engine.budget_client.deduct_funds.assert_not_called()

        # Should NOT call audit (skipped)
        mock_engine.audit_client.log_transaction.assert_not_called()


def test_zero_cost_deduction(smart_client: SmartClient, mock_engine: MagicMock) -> None:
    """
    Verify that 0 cost is handled correctly.
    """
    messages = [{"role": "user", "content": "Hello"}]

    with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 0
        mock_response.usage.completion_tokens = 0
        mock_completion.return_value = mock_response

        mock_engine.budget_client.check_allowance.return_value = True

        smart_client.chat.completions.create(messages, user="test_user")

        # Cost should be 0.0
        mock_engine.budget_client.deduct_funds.assert_called_once_with(user_id="test_user", amount=0.0)


def test_fail_open_missing_usage_logs_error(smart_client: SmartClient, mock_engine: MagicMock) -> None:
    """
    Verify that if response.usage is missing in fail-open, the response is returned, and accounting is skipped safely.
    """
    messages = [{"role": "user", "content": "Hello"}]

    with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
        # Force fail-open
        from typing import cast
        from unittest.mock import MagicMock

        cast(MagicMock, smart_client.chat.completions.router.route).side_effect = RuntimeError("No models")

        mock_response = MagicMock()
        # Delete usage attribute to simulate missing data
        del mock_response.usage

        mock_completion.return_value = mock_response
        mock_engine.budget_client.check_allowance.return_value = True

        response = smart_client.chat.completions.create(messages, user="test_user")

        # Should return the response successfully
        assert response == mock_response

        # Should NOT call deduct_funds (skipped due to error in cost calc)
        mock_engine.budget_client.deduct_funds.assert_not_called()

        # Should NOT call audit (skipped)
        mock_engine.audit_client.log_transaction.assert_not_called()
