# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from unittest.mock import Mock

import pytest
from coreason_identity.models import UserContext

from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier, RoutingContext
from coreason_arbitrage.registry import ModelRegistry
from coreason_arbitrage.router import Router

# --- Fixtures ---


@pytest.fixture
def mock_registry() -> Mock:
    registry = Mock(spec=ModelRegistry)

    # Standard models
    t1 = ModelDefinition(
        id="tier1-model", provider="test", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0.01, cost_per_1k_output=0.01
    )
    t2 = ModelDefinition(
        id="tier2-model", provider="test", tier=ModelTier.TIER_2_SMART, cost_per_1k_input=0.05, cost_per_1k_output=0.05
    )

    # Domain specific model
    oncology_model = ModelDefinition(
        id="oncology-llama-3",
        provider="test",
        tier=ModelTier.TIER_2_SMART,
        cost_per_1k_input=0.06,
        cost_per_1k_output=0.06,
        domain="oncology",
    )

    def list_models_side_effect(tier: ModelTier | None = None, domain: str | None = None) -> list[ModelDefinition]:
        all_models = [t1, t2, oncology_model]
        filtered = all_models
        if tier:
            filtered = [m for m in filtered if m.tier == tier]
        if domain:
            filtered = [m for m in filtered if m.domain and m.domain.lower() == domain.lower()]
        return filtered

    registry.list_models.side_effect = list_models_side_effect
    return registry


@pytest.fixture
def mock_budget_client() -> Mock:
    client = Mock(spec=BudgetClient)
    client.get_remaining_budget_percentage.return_value = 1.0
    return client


@pytest.fixture
def router(mock_registry: Mock, mock_budget_client: Mock) -> Router:
    return Router(registry=mock_registry, budget_client=mock_budget_client)


# --- Tests ---


def test_route_prioritizes_domain_model(router: Router, user_context: UserContext) -> None:
    """
    Test that if a domain-specific model exists and is requested,
    it is returned regardless of complexity.
    """
    # Context with 'oncology' domain. Complexity is low (0.1), which normally routes to Tier 1.
    # However, we have a specialized oncology model (Tier 2).
    context = RoutingContext(complexity=0.1, domain="oncology")

    model = router.route(context, user_context=user_context)

    assert model.id == "oncology-llama-3"
    assert model.domain == "oncology"


def test_route_domain_model_case_insensitive(router: Router, user_context: UserContext) -> None:
    """
    Test that domain matching is case insensitive.
    """
    context = RoutingContext(complexity=0.1, domain="OnCoLoGy")

    model = router.route(context, user_context=user_context)

    assert model.id == "oncology-llama-3"


def test_route_fallback_if_domain_model_not_found(router: Router, user_context: UserContext) -> None:
    """
    Test that if no model exists for the requested domain,
    it falls back to standard complexity-based routing.
    """
    # Domain 'legal' has no specific model. Complexity 0.1 -> Tier 1.
    context = RoutingContext(complexity=0.1, domain="legal")

    model = router.route(context, user_context=user_context)

    assert model.id == "tier1-model"
    assert model.tier == ModelTier.TIER_1_FAST


def test_route_fallback_if_domain_model_unhealthy(router: Router, mock_registry: Mock, user_context: UserContext) -> None:
    """
    Test that if the domain model is unhealthy, it falls back to standard routing.
    """
    # Override registry to return an unhealthy oncology model
    unhealthy_oncology = ModelDefinition(
        id="oncology-llama-3",
        provider="test",
        tier=ModelTier.TIER_2_SMART,
        cost_per_1k_input=0.06,
        cost_per_1k_output=0.06,
        domain="oncology",
        is_healthy=False,
    )
    t1 = ModelDefinition(
        id="tier1-model", provider="test", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0.01, cost_per_1k_output=0.01
    )

    def list_models_side_effect(tier: ModelTier | None = None, domain: str | None = None) -> list[ModelDefinition]:
        all_models = [t1, unhealthy_oncology]
        filtered = all_models
        if tier:
            filtered = [m for m in filtered if m.tier == tier]
        if domain:
            filtered = [m for m in filtered if m.domain and m.domain.lower() == domain.lower()]
        return filtered

    mock_registry.list_models.side_effect = list_models_side_effect

    context = RoutingContext(complexity=0.1, domain="oncology")

    # Should skip unhealthy oncology model and fallback to Tier 1 (due to low complexity)
    model = router.route(context, user_context=user_context)

    assert model.id == "tier1-model"


def test_route_safety_critical_overrides_complexity_but_checked_after_domain_priority(router: Router, user_context: UserContext) -> None:
    """
    If domain is 'safety_critical', it triggers Tier 3 in standard logic.
    But if there was a specific model for 'safety_critical' domain registered,
    the domain priority check should catch it first.

    However, 'safety_critical' is usually a flag for Tier 3, not necessarily a custom model name.
    If no custom model is registered for 'safety_critical', it falls through to standard logic
    which checks for 'safety_critical' domain string to enforce Tier 3.
    """
    # Ensure no model registered with domain='safety_critical'
    # Request: domain='safety_critical', complexity=0.1
    # Expect: Tier 3 generic model (because of fallback logic)

    context = RoutingContext(complexity=0.1, domain="safety_critical")

    # We need to make sure our mock registry returns a Tier 3 model for the fallback to work
    t3 = ModelDefinition(
        id="tier3-model",
        provider="test",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.1,
        cost_per_1k_output=0.1,
    )

    # Update mock to include T3
    # We access the mock through the router's registry which we know is a mock in this test setup
    registry_mock = router.registry
    assert isinstance(registry_mock, Mock)

    original_side_effect = registry_mock.list_models.side_effect

    def new_side_effect(tier: ModelTier | None = None, domain: str | None = None) -> list[ModelDefinition]:
        res: list[ModelDefinition] = original_side_effect(tier, domain)
        if tier == ModelTier.TIER_3_REASONING:
            res.append(t3)
        return res

    registry_mock.list_models.side_effect = new_side_effect

    model = router.route(context, user_context=user_context)
    assert model.tier == ModelTier.TIER_3_REASONING
