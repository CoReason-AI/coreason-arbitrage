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


def test_domain_to_generic_fallback() -> None:
    """
    Test 1: Domain to Generic Fallback.
    Context: Prompt has domain "medical".
    Registry: 1 Medical Model (Tier 3), 1 Generic Model (Tier 3).
    Behavior:
        - Attempt 1: Router picks Medical Model. Execution Fails (503). Medical Model excluded.
        - Attempt 2: Router sees Medical Model excluded. healthy_domain_candidates becomes empty.
                     Router proceeds to Step 4 (Generic). Picks Generic Model. Execution Succeeds.
    """
    engine = ArbitrageEngine()
    engine.registry.clear()

    # Medical Model
    medical_model = ModelDefinition(
        id="medical-model",
        provider="med-provider",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        is_healthy=True,
        domain="medical",
    )
    # Generic Model
    generic_model = ModelDefinition(
        id="generic-model",
        provider="gen-provider",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        is_healthy=True,
    )

    engine.registry.register_model(medical_model)
    engine.registry.register_model(generic_model)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    with engine._lock:
        engine.budget_client = mock_budget

    client = engine.get_client()

    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 10

    def side_effect(model: str, **kwargs: Any) -> Any:
        if model == "medical-model":
            raise ServiceUnavailableError("Medical Down", model="medical-model", llm_provider="med-provider")
        elif model == "generic-model":
            return mock_response
        else:
            raise ValueError(f"Unknown model: {model}")

    with patch("coreason_arbitrage.smart_client.acompletion", side_effect=side_effect) as mock_completion:
        # Prompt "clinical" triggers 'medical' domain and Tier 3 (via complexity or just domain match)
        # Gatekeeper: medical -> medical domain.
        # Router: Tier 3 target.
        messages = [{"role": "user", "content": "Analyze this clinical data."}]

        response = client.chat.completions.create(messages=messages)

        assert response == mock_response
        # Verify call order
        mock_completion.assert_any_call(model="medical-model", messages=messages)
        mock_completion.assert_any_call(model="generic-model", messages=messages)
        assert mock_completion.call_count == 2


def test_non_critical_error_does_not_exclude() -> None:
    """
    Test 2: Non-Critical Error Persistence.
    Setup: 1 Model.
    Action: Mock BadRequestError (400).
    Expectation: Retries same model 3 times (does not exclude).
    """
    engine = ArbitrageEngine()
    engine.registry.clear()

    model = ModelDefinition(
        id="bad-model",
        provider="bad-provider",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
        is_healthy=True,
    )
    engine.registry.register_model(model)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    with engine._lock:
        engine.budget_client = mock_budget

    client = engine.get_client()

    with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
        # Side effect: Always raise BadRequestError
        mock_completion.side_effect = BadRequestError("Bad Request", model="bad-model", llm_provider="bad-provider")

        with patch("coreason_arbitrage.smart_client.logger") as mock_logger:
            # We expect it to retry 3 times then Fail Open
            # Wait, if it fails open, it calls fallback.
            # But we want to check that it *retried* the same model 3 times.
            # And did NOT exclude it.

            # We can check exclusion by spying on Router.route, or just inference from calls.
            # If it excluded, Router would raise "No healthy models" on 2nd attempt.
            # If it didn't exclude, Router returns same model.

            # Fail Open will happen.
            with pytest.raises(BadRequestError):
                client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

            # Verify 3 attempts on 'bad-model' + 1 attempt on 'fallback'
            # 3 attempts (MAX_RETRIES=3)
            # Actually, the loop runs MAX_RETRIES times.
            # Attempt 0: calls bad-model -> BadRequest -> continue
            # Attempt 1: calls bad-model -> BadRequest -> continue
            # Attempt 2: calls bad-model -> BadRequest -> continue
            # Loop ends. Fail Open.
            # Fail Open calls 'azure/gpt-4o'.

            # So we expect 3 calls to 'bad-model'.
            calls_to_bad = [call for call in mock_completion.call_args_list if call.kwargs.get("model") == "bad-model"]
            assert len(calls_to_bad) == 3

            # Verify warning about exclusion was NOT logged
            for call in mock_logger.warning.call_args_list:
                msg = str(call)
                if "Excluding from retry" in msg:
                    assert "bad-provider" not in msg


def test_full_exhaustion_fail_open() -> None:
    """
    Test 3: Full Exhaustion to Fail-Open.
    Setup: 2 Generic Models.
    Action: Both fail with 503.
    Expectation: Call A (fail, exclude), Call B (fail, exclude),
                 Router raises Error (No models), Fail-Open triggers.
    """
    engine = ArbitrageEngine()
    engine.registry.clear()

    model_a = ModelDefinition(
        id="model-a",
        provider="provider-a",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
    )
    model_b = ModelDefinition(
        id="model-b",
        provider="provider-b",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
    )
    engine.registry.register_model(model_a)
    engine.registry.register_model(model_b)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    with engine._lock:
        engine.budget_client = mock_budget

    client = engine.get_client()

    with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
        # Both fail
        def side_effect(model: str, **kwargs: Any) -> Any:
            if model == "model-a":
                raise ServiceUnavailableError("Service Down", model="model-a", llm_provider="provider-a")
            elif model == "model-b":
                raise ServiceUnavailableError("Service Down", model="model-b", llm_provider="provider-b")
            elif model == "azure/gpt-4o":  # Fallback
                return MagicMock()
            else:
                return MagicMock()

        mock_completion.side_effect = side_effect

        # Run
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

        # Verify calls: A and B should both be called.
        # Order depends on dict iteration, but both should be hit before fail open.
        calls = [
            c.kwargs.get("model")
            for c in mock_completion.call_args_list
            if c.kwargs.get("model") in ["model-a", "model-b"]
        ]
        assert "model-a" in calls
        assert "model-b" in calls
        assert len(calls) == 2

        # Verify fallback called
        mock_completion.assert_called_with(model="azure/gpt-4o", messages=[{"role": "user", "content": "hi"}])
