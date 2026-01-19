# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import BadRequestError, ServiceUnavailableError

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier


@pytest.fixture
def configured_engine() -> ArbitrageEngine:
    engine = ArbitrageEngine()
    engine.registry.clear()

    # Register two models in same tier
    model1 = ModelDefinition(
        id="provider1/model",
        provider="provider1",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        is_healthy=True,
    )
    model2 = ModelDefinition(
        id="provider2/model",
        provider="provider2",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        is_healthy=True,
    )
    engine.registry.register_model(model1)
    engine.registry.register_model(model2)

    # Configure budget
    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    with engine._lock:
        engine.budget_client = mock_budget

    return engine


def test_failover_specificity_transient_error(configured_engine: ArbitrageEngine) -> None:
    """
    Verifies that a ServiceUnavailableError triggers failover and LB recording.
    """
    client = configured_engine.get_client()

    # Spy on LoadBalancer
    with patch.object(
        configured_engine.load_balancer, "record_failure", wraps=configured_engine.load_balancer.record_failure
    ) as mock_lb_record:
        # Mock completion: First call fails (provider1), Second call succeeds (provider2)

        mock_success = MagicMock()
        mock_success.usage.prompt_tokens = 10
        mock_success.usage.completion_tokens = 10

        def side_effect(model: str, **kwargs: Any) -> Any:
            if "provider1" in model:
                raise ServiceUnavailableError("Down", model=model, llm_provider="provider1")
            return mock_success

        with patch("coreason_arbitrage.smart_client.completion", side_effect=side_effect) as mock_completion:
            # We force Router to pick provider1 first by explicitly excluding provider2?
            # Or we can just let Router pick. If it picks provider2 first, we don't test failover.
            # So we should exclude provider2 initially? No, we want failover.
            # We can mock Router.route to return provider1 then provider2.

            with patch.object(client.chat.completions.router, "route") as mock_route:
                # Setup models
                m1 = configured_engine.registry.get_model("provider1/model")
                m2 = configured_engine.registry.get_model("provider2/model")
                assert m1 and m2

                # First call returns m1, Second call returns m2
                mock_route.side_effect = [m1, m2, m2]

                messages = [{"role": "user", "content": "Analyze this."}]
                response = client.chat.completions.create(messages=messages)

                assert response == mock_success

                # Verify Failover Happened
                assert mock_completion.call_count == 2

                # Verify LB recorded failure for provider1
                mock_lb_record.assert_called_with("provider1")

                # Verify Router was called with excluded_providers=["provider1"] in the second call
                # Check call args of the SECOND call to route
                # mock_route.call_args_list[0] -> initial call (empty exclusions)
                # mock_route.call_args_list[1] -> retry call

                args, kwargs = mock_route.call_args_list[1]
                assert "excluded_providers" in kwargs
                assert "provider1" in kwargs["excluded_providers"]


def test_failover_specificity_client_error(configured_engine: ArbitrageEngine) -> None:
    """
    Verifies that a BadRequestError does NOT trigger LB recording or exclusion.
    """
    client = configured_engine.get_client()

    with patch.object(
        configured_engine.load_balancer, "record_failure", wraps=configured_engine.load_balancer.record_failure
    ) as mock_lb_record:
        # Mock completion: Always fails with BadRequestError
        def side_effect(model: str, **kwargs: Any) -> Any:
            raise BadRequestError("Bad Request", model=model, llm_provider="provider1")

        with patch("coreason_arbitrage.smart_client.completion", side_effect=side_effect) as mock_completion:
            with patch.object(client.chat.completions.router, "route") as mock_route:
                m1 = configured_engine.registry.get_model("provider1/model")
                assert m1
                mock_route.return_value = m1

                messages = [{"role": "user", "content": "Analyze this."}]

                # Expect Fail Open or Exception?
                # It will loop 3 times then Fail Open.
                # Fail Open will try fallback model (which we didn't mock completion for specifically,
                # so side_effect runs).
                # If side_effect raises BadRequestError, eventually it raises out.

                with pytest.raises(BadRequestError):
                    client.chat.completions.create(messages=messages)

                # Verify LB did NOT record failure
                mock_lb_record.assert_not_called()

                # Verify Router was called multiple times but WITHOUT provider1 in exclusions
                # We expect 3 calls (attempts 0, 1, 2)
                assert mock_route.call_count >= 3
                assert mock_completion.called

                for call in mock_route.call_args_list:
                    args, kwargs = call
                    # excluded_providers should be empty or not contain provider1
                    excluded = kwargs.get("excluded_providers", [])
                    assert "provider1" not in excluded
