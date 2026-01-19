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
from litellm.exceptions import BadRequestError, RateLimitError

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier


@pytest.fixture
def mock_engine() -> ArbitrageEngine:
    engine = ArbitrageEngine()
    # Clear state
    engine.registry.clear()
    if hasattr(engine, "load_balancer"):
        engine.load_balancer._failures.clear()
        engine.load_balancer._cooldown_until.clear()

    # Configure with mocks
    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    engine.configure(mock_budget, MagicMock(), MagicMock())
    return engine


def test_smart_client_retry_loop_accumulates_failures(mock_engine: ArbitrageEngine) -> None:
    """
    Unit test to verify that:
    1. Critical errors (RateLimitError) trigger record_failure.
    2. Failed providers are added to 'excluded_providers' in subsequent routing calls.
    """
    client = mock_engine.get_client()

    # Define models
    model_a = ModelDefinition(
        id="model-a", provider="provider-a", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0, cost_per_1k_output=0
    )
    model_b = ModelDefinition(
        id="model-b", provider="provider-b", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0, cost_per_1k_output=0
    )

    # Spy on LoadBalancer using patch.object to satisfy mypy
    with patch.object(
        mock_engine.load_balancer, "record_failure", wraps=mock_engine.load_balancer.record_failure
    ) as mock_record_failure:
        # Mock Router.route to return A then B
        client.chat.completions.router.route = MagicMock(side_effect=[model_a, model_b])

        # Mock completion to fail for A, succeed for B
        with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
            mock_completion.side_effect = [
                RateLimitError("Limit Hit", model="model-a", llm_provider="provider-a"),
                MagicMock(usage=MagicMock(prompt_tokens=10, completion_tokens=10)),
            ]

            client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

            # Verify LoadBalancer recorded failure for provider-a
            mock_record_failure.assert_called_with("provider-a")

            # Verify Router was called with exclusions
            # Call 1: empty exclusion
            # Call 2: provider-a excluded
            route_calls = client.chat.completions.router.route.call_args_list
            assert len(route_calls) == 2

            # Check call 1 args
            args1, kwargs1 = route_calls[0]
            assert kwargs1["excluded_providers"] == []

            # Check call 2 args
            args2, kwargs2 = route_calls[1]
            assert "provider-a" in kwargs2["excluded_providers"]


def test_smart_client_does_not_record_failure_for_bad_request(mock_engine: ArbitrageEngine) -> None:
    """
    Unit test to verify that:
    1. Non-critical errors (BadRequestError) do NOT trigger record_failure.
    2. Provider is NOT excluded in subsequent retry.
    """
    client = mock_engine.get_client()

    model_a = ModelDefinition(
        id="model-a", provider="provider-a", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0, cost_per_1k_output=0
    )

    # Spy on LoadBalancer
    with patch.object(
        mock_engine.load_balancer, "record_failure", wraps=mock_engine.load_balancer.record_failure
    ) as mock_record_failure:
        # Mock Router always returning model_a
        client.chat.completions.router.route = MagicMock(return_value=model_a)

        # Mock completion to fail with BadRequest twice, then succeed (or fail open, but we check calls)
        with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
            mock_completion.side_effect = [
                BadRequestError("Bad Request", model="model-a", llm_provider="provider-a"),
                MagicMock(usage=MagicMock(prompt_tokens=10, completion_tokens=10)),
            ]

            client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

            # Verify LoadBalancer did NOT record failure
            mock_record_failure.assert_not_called()

            # Verify Router calls did NOT exclude provider-a
            route_calls = client.chat.completions.router.route.call_args_list
            assert len(route_calls) == 2  # 1st fail, 2nd succeed

            # Check call 2 args
            args2, kwargs2 = route_calls[1]
            assert "provider-a" not in kwargs2["excluded_providers"]
