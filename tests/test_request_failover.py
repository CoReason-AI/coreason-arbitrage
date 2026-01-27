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

from coreason_identity.models import UserContext
from litellm.exceptions import ServiceUnavailableError

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier


def test_request_scoped_failover() -> None:
    """
    Test that SmartClient excludes a provider that fails during the request
    and retries with a different provider in the same tier immediately.
    """
    engine = ArbitrageEngine()

    # 1. Setup Registry with 2 models in same Tier
    azure_model = ModelDefinition(
        id="azure/gpt-4o",
        provider="azure",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        is_healthy=True,
    )
    aws_model = ModelDefinition(
        id="aws/claude-3-opus",
        provider="aws",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        is_healthy=True,
    )

    # Manually register models (bypassing configure for simplicity)
    engine.registry.clear()
    engine.registry.register_model(azure_model)
    engine.registry.register_model(aws_model)

    # 2. Configure Mock Budget (so it doesn't fail pre-flight)
    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    # We need to set the budget client on the engine so SmartClient sees it
    with engine._lock:
        engine.budget_client = mock_budget

    client = engine.get_client()

    # 3. Mock completion to fail for Azure, succeed for AWS
    # We need to control the order. Since registry order is not guaranteed,
    # we need to ensure the Router picks Azure first OR handle both orders.
    # However, since we are testing failover, if it picks AWS first,
    # the test passes trivially but doesn't test failover.
    # To force failover test, we can make Azure fail. If Router picks Azure first
    # -> Fail -> Retry AWS -> Success.
    # If Router picks AWS first -> Success.
    # So we need to ensure Router *prefers* Azure or just mock based on model input.

    # Let's mock completion using side_effect with a function
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 10

    def side_effect(model: str, **kwargs: Any) -> Any:
        if model == "azure/gpt-4o":
            # Simulate 503
            raise ServiceUnavailableError("Azure is down", model="azure/gpt-4o", llm_provider="azure")
        elif model == "aws/claude-3-opus":
            return mock_response
        else:
            raise ValueError(f"Unknown model: {model}")

    # 4. Patch completion and Router
    # Note: We need to patch Router.route only if we want to spy on it,
    # but we are testing that SmartClient *calls* Router with exclusion.
    # So we should rely on the real Router logic (which we will modify).
    # Ideally, we want to ensure Azure is picked first.
    # Current Router picks first healthy model.
    # Let's force registry order by clearing and adding Azure first.
    engine.registry.clear()
    engine.registry.register_model(azure_model)  # Azure added first
    engine.registry.register_model(aws_model)  # AWS added second
    # Note: Python dicts preserve insertion order in 3.7+, so listing models should yield Azure first.

    uc = MagicMock(spec=UserContext)
    uc.user_id = "user1"
    uc.groups = []

    with patch("coreason_arbitrage.smart_client.acompletion", side_effect=side_effect) as mock_completion:
        # Run
        # Prompt "Analyze" triggers Tier 3
        messages = [{"role": "user", "content": "Analyze this data."}]

        # We expect SmartClient to pass excluded_providers to router.
        # But we haven't implemented that yet.
        # This test will likely fail with "ServiceUnavailableError" or infinite loop on Azure
        # (if retry logic just retries Azure 3 times).

        response = client.chat.completions.create(messages=messages, user_context=uc)

        # Assertions
        assert response == mock_response

        # Verify Azure was called
        mock_completion.assert_any_call(model="azure/gpt-4o", messages=messages)
        # Verify AWS was called
        mock_completion.assert_any_call(model="aws/claude-3-opus", messages=messages)

        # Verify call count: 1 fail (Azure) + 1 success (AWS) = 2 calls
        # Unless it retried Azure multiple times.
        # Without exclusion, it would retry Azure 3 times (Max Retries) then fail open.
        # So if we see 2 calls, it means it switched.
        assert mock_completion.call_count == 2
