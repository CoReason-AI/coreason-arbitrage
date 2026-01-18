# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import ServiceUnavailableError

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import AuditClient, BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier


@pytest.fixture
def configured_engine() -> ArbitrageEngine:
    engine = ArbitrageEngine()
    engine.registry.clear()

    # Check if load_balancer is initialized before accessing it.
    if hasattr(engine, "load_balancer"):
        engine.load_balancer._failures.clear()
        engine.load_balancer._cooldown_until.clear()

    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    mock_budget.get_remaining_budget_percentage.return_value = 0.5

    mock_audit = MagicMock(spec=AuditClient)
    mock_foundry = MagicMock()

    engine.configure(mock_budget, mock_audit, mock_foundry)
    return engine


def test_cascading_failover(configured_engine: ArbitrageEngine) -> None:
    """
    Test that if the primary model fails, the client automatically retries
    with a secondary model (after the first one is marked unhealthy).
    """
    # Setup: 2 Models in TIER_1
    model_a = ModelDefinition(
        id="azure-model",
        provider="azure",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
    )
    model_b = ModelDefinition(
        id="aws-model",
        provider="aws",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
    )

    configured_engine.registry.register_model(model_a)
    configured_engine.registry.register_model(model_b)

    client = configured_engine.get_client()
    messages = [{"role": "user", "content": "hello"}]

    # Mock completion to fail for Azure, succeed for AWS
    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:

        def side_effect(model: str, messages: Any, **kwargs: Any) -> Any:
            if model == "azure-model":
                raise ServiceUnavailableError("Azure 503 Service Unavailable", model=model, llm_provider="azure")
            elif model == "aws-model":
                return MagicMock(id="aws-response")
            return MagicMock()

        mock_completion.side_effect = side_effect

        with patch("coreason_arbitrage.load_balancer.FAILURE_THRESHOLD", 0):
            # Threshold 0 means 1 failure trips it (>0).

            response = client.chat.completions.create(messages=messages, user="test_user")

            assert response.id == "aws-response"

            # Verify Azure marked unhealthy
            assert not configured_engine.load_balancer.is_provider_healthy("azure")
            # Verify AWS healthy
            assert configured_engine.load_balancer.is_provider_healthy("aws")

            # Verify completion called at least twice (Azure then AWS)
            assert mock_completion.call_count >= 2


def test_total_outage(configured_engine: ArbitrageEngine) -> None:
    """
    Test that if all providers fail, the client raises an exception after retries.
    """
    model_a = ModelDefinition(
        id="azure-model",
        provider="azure",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
    )
    configured_engine.registry.register_model(model_a)

    client = configured_engine.get_client()
    messages = [{"role": "user", "content": "hello"}]

    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        mock_completion.side_effect = ServiceUnavailableError(
            "Global Outage", model="azure-model", llm_provider="azure"
        )

        with patch("coreason_arbitrage.load_balancer.FAILURE_THRESHOLD", 0):
            # The exception might be "No healthy models" because the model gets marked unhealthy
            # and next retry fails routing.
            with pytest.raises(Exception) as excinfo:
                client.chat.completions.create(messages=messages, user="test_user")

            # Verify it's either the outage or routing error
            assert "Global Outage" in str(excinfo.value) or "No healthy models" in str(excinfo.value)

            # Should have marked unhealthy
            assert not configured_engine.load_balancer.is_provider_healthy("azure")


def test_concurrency_stress(configured_engine: ArbitrageEngine) -> None:
    """
    Test concurrent requests updating LoadBalancer state.
    """
    model_a = ModelDefinition(
        id="azure-model",
        provider="azure",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
    )
    configured_engine.registry.register_model(model_a)

    client = configured_engine.get_client()

    # Move patch outside threads to avoid race conditions with global module patching
    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        mock_completion.side_effect = ServiceUnavailableError(
            "Concurrency Error", model="azure-model", llm_provider="azure"
        )

        def worker() -> None:
            try:
                # We simulate failure
                client.chat.completions.create(messages=[{"role": "user", "content": "stress"}])
            except Exception:
                pass

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # Azure should definitely be unhealthy now
    assert not configured_engine.load_balancer.is_provider_healthy("azure")
