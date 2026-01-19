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
def complex_engine() -> ArbitrageEngine:
    engine = ArbitrageEngine()
    engine.registry.clear()

    # Register 3 providers for complex scenarios
    for i in range(1, 4):
        model = ModelDefinition(
            id=f"provider{i}/gpt-4",
            provider=f"provider{i}",
            tier=ModelTier.TIER_3_REASONING,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            is_healthy=True,
        )
        engine.registry.register_model(model)

    # Configure budget
    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    mock_budget.get_remaining_budget_percentage.return_value = 0.5  # Return float
    with engine._lock:
        engine.budget_client = mock_budget

    return engine


def test_total_meltdown_fail_open(complex_engine: ArbitrageEngine) -> None:
    """
    Scenario A: All providers fail with ServiceUnavailableError.
    Expected:
    - Retries exhaust all providers (or MAX_RETRIES).
    - Providers are excluded sequentially.
    - Ultimately triggers Fail-Open.
    """
    client = complex_engine.get_client()

    # Spy on LB
    with patch.object(complex_engine.load_balancer, "record_failure") as mock_lb_record:
        # Mock completion to always raise ServiceUnavailable
        def side_effect(model: str, **kwargs: Any) -> Any:
            # Check if this is the fallback model
            # SmartClient defaults fallback to "azure/gpt-4o"
            if model == "azure/gpt-4o":
                return MagicMock(usage=MagicMock(prompt_tokens=10, completion_tokens=10))

            # Identify provider from model ID
            provider = model.split("/")[0]
            raise ServiceUnavailableError("Down", model=model, llm_provider=provider)

        with patch("coreason_arbitrage.smart_client.completion", side_effect=side_effect) as mock_completion:
            messages = [{"role": "user", "content": "Analyze."}]

            # We expect a success eventually (Fail Open Success) or return of the mock
            response = client.chat.completions.create(messages=messages)

            # Assert response is valid (Fail Open worked)
            assert response is not None

            # Verify failures recorded. SmartClient MAX_RETRIES is 3.
            # So it tries 3 times.
            # Attempt 1: Provider X -> Fails -> LB Record X -> Exclude X
            # Attempt 2: Provider Y -> Fails -> LB Record Y -> Exclude Y
            # Attempt 3: Provider Z -> Fails -> LB Record Z -> Exclude Z
            # Fail Open triggered.

            # NOTE: If we have 3 providers and MAX_RETRIES=3, we expect 3 LB records.
            # If MAX_RETRIES < 3, we expect fewer. Defaults to 3.
            assert mock_lb_record.call_count == 3
            assert mock_completion.called

            # Check arguments
            providers_recorded = [args[0] for args, _ in mock_lb_record.call_args_list]
            # Since Router picks generically, we can't guarantee order, but they should be unique providers
            # unless Router picks same one (which it shouldn't if exclusion works).
            assert len(set(providers_recorded)) == 3


def test_mixed_failure_types(complex_engine: ArbitrageEngine) -> None:
    """
    Scenario B:
    - Attempt 1: Provider 1 (ServiceUnavailable) -> Should Exclude
    - Attempt 2: Provider 2 (BadRequest) -> Should NOT Exclude
    - Attempt 3: Provider 2 (BadRequest) -> Loops on same?
    """
    client = complex_engine.get_client()

    with patch.object(complex_engine.load_balancer, "record_failure") as mock_lb_record:
        # We need to control the sequence of errors based on the model selected.
        # But we don't know which model Router picks first.
        # We can spy on Router to see which one it picks, or use side_effect that adapts.

        # Let's mock Router to be deterministic
        with patch.object(client.chat.completions.router, "route") as mock_route:
            p1 = complex_engine.registry.get_model("provider1/gpt-4")
            p2 = complex_engine.registry.get_model("provider2/gpt-4")

            # Force Sequence: P1, P2, P2, P2
            mock_route.side_effect = [p1, p2, p2, p2]

            def side_effect(model: str, **kwargs: Any) -> Any:
                if "provider1" in model:
                    raise ServiceUnavailableError("Down", model=model, llm_provider="provider1")
                elif "provider2" in model:
                    raise BadRequestError("Bad Prompt", model=model, llm_provider="provider2")
                return MagicMock()

            with patch("coreason_arbitrage.smart_client.completion", side_effect=side_effect):
                messages = [{"role": "user", "content": "Bad Request."}]

                # It should eventually raise BadRequestError (if it propagates last exception)
                # or Fail Open if it catches everything?
                # SmartClient implementation catches Exception.
                # If it's BadRequestError (not retriable), it does NOT exclude.
                # It continues the loop.
                # It loops MAX_RETRIES times.
                # Then Fails Open.
                # Fail Open calls "azure/gpt-4o". This succeeds (returns MagicMock).

                # So response will be the fallback success.
                response = client.chat.completions.create(messages=messages)
                assert response is not None

                # Check LB Record: Should only record P1
                mock_lb_record.assert_called_once_with("provider1")

                # Check Exclusions passed to Router
                # Call 1: Exclude []
                # Call 2 (after P1 fail): Exclude [P1]
                # Call 3 (after P2 fail): Exclude [P1] (P2 NOT added)

                _, kwargs2 = mock_route.call_args_list[1]
                assert "provider1" in kwargs2["excluded_providers"]
                assert "provider2" not in kwargs2["excluded_providers"]

                _, kwargs3 = mock_route.call_args_list[2]
                assert "provider1" in kwargs3["excluded_providers"]
                assert "provider2" not in kwargs3["excluded_providers"]
