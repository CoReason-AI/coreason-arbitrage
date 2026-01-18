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

from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier, RoutingContext
from coreason_arbitrage.registry import ModelRegistry
from coreason_arbitrage.router import Router


@pytest.fixture
def mock_registry() -> Mock:
    registry = Mock(spec=ModelRegistry)
    return registry


@pytest.fixture
def mock_budget_client() -> Mock:
    client = Mock(spec=BudgetClient)
    return client


@pytest.fixture
def router(mock_registry: Mock, mock_budget_client: Mock) -> Router:
    return Router(registry=mock_registry, budget_client=mock_budget_client)


def test_economy_mode_boundary(router: Router) -> None:
    """
    Test the boundary condition for Economy Mode.
    Budget < 10% (0.10) triggers downgrade.
    Budget == 10% (0.10) should NOT trigger downgrade.
    """
    # Setup
    context = RoutingContext(complexity=0.5, domain=None)  # Tier 2 normally

    # Case 1: Budget 0.10 (10%) -> No Downgrade
    assert isinstance(router.budget_client.get_remaining_budget_percentage, Mock)
    router.budget_client.get_remaining_budget_percentage.return_value = 0.10

    # We need the registry to return a Tier 2 model
    tier2_model = ModelDefinition(
        id="tier2-model", provider="test", tier=ModelTier.TIER_2_SMART, cost_per_1k_input=0.05, cost_per_1k_output=0.05
    )
    assert isinstance(router.registry.list_models, Mock)
    router.registry.list_models.return_value = [tier2_model]

    model = router.route(context, user_id="user1")
    assert model.tier == ModelTier.TIER_2_SMART

    # Verify we asked for Tier 2
    args, _ = router.registry.list_models.call_args
    assert args == ()
    kwargs = router.registry.list_models.call_args.kwargs
    assert kwargs.get("tier") == ModelTier.TIER_2_SMART

    # Case 2: Budget 0.099 (9.9%) -> Downgrade to Tier 1
    router.budget_client.get_remaining_budget_percentage.return_value = 0.099

    tier1_model = ModelDefinition(
        id="tier1-model", provider="test", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0.01, cost_per_1k_output=0.01
    )
    router.registry.list_models.return_value = [tier1_model]

    model = router.route(context, user_id="user1")
    assert model.tier == ModelTier.TIER_1_FAST

    # Verify we asked for Tier 1
    kwargs = router.registry.list_models.call_args.kwargs
    assert kwargs.get("tier") == ModelTier.TIER_1_FAST


def test_domain_priority_bypasses_economy(router: Router) -> None:
    """
    Test that if a domain-specific model exists, it is selected even if
    Economy Mode would have downgraded the generic tier.

    Scenario:
    - Complexity: Tier 2 (0.5)
    - Budget: Low (0.05) -> Economy Mode tries to downgrade to Tier 1.
    - Domain: "specialized"
    - Available Models:
        - "special-tier2" (Domain: specialized, Tier 2)

    Expected:
    - Target Tier becomes Tier 1 (due to economy).
    - Domain check looks for specialized models.
    - Finds "special-tier2".
    - "special-tier2" is Tier 2, so it does NOT match Target Tier 1.
    - Soft Fallback picks "special-tier2" anyway because it's the only healthy domain model.
    """
    context = RoutingContext(complexity=0.5, domain="specialized")

    assert isinstance(router.budget_client.get_remaining_budget_percentage, Mock)
    router.budget_client.get_remaining_budget_percentage.return_value = 0.05

    special_model = ModelDefinition(
        id="special-tier2",
        provider="test",
        tier=ModelTier.TIER_2_SMART,
        cost_per_1k_input=0.05,
        cost_per_1k_output=0.05,
        domain="specialized",
    )

    # Setup mock for list_models(domain="specialized")
    # And generic fallback calls
    def list_models_side_effect(tier: ModelTier | None = None, domain: str | None = None) -> list[ModelDefinition]:
        if domain == "specialized":
            return [special_model]
        return []

    assert isinstance(router.registry.list_models, Mock)
    router.registry.list_models.side_effect = list_models_side_effect

    model = router.route(context, user_id="user1")

    # It should return the domain model despite Economy Mode
    assert model.id == "special-tier2"
    assert model.tier == ModelTier.TIER_2_SMART


def test_domain_multiple_tiers_fallback(router: Router) -> None:
    """
    Test selection behavior when multiple domain models exist but none match
    the target tier.

    Scenario:
    - Complexity: Tier 3 (0.9)
    - Domain: "code"
    - Available Domain Models:
        1. "code-small" (Tier 1)
        2. "code-medium" (Tier 2)

    Expected:
    - Target Tier: Tier 3.
    - Domain Check finds [code-small, code-medium].
    - No match for Tier 3.
    - Fallback: Returns the first one in the list (code-small).
    """
    context = RoutingContext(complexity=0.9, domain="code")

    assert isinstance(router.budget_client.get_remaining_budget_percentage, Mock)
    router.budget_client.get_remaining_budget_percentage.return_value = 1.0

    code_small = ModelDefinition(
        id="code-small",
        provider="test",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.01,
        domain="code",
    )
    code_medium = ModelDefinition(
        id="code-medium",
        provider="test",
        tier=ModelTier.TIER_2_SMART,
        cost_per_1k_input=0.05,
        cost_per_1k_output=0.05,
        domain="code",
    )

    def list_models_side_effect(tier: ModelTier | None = None, domain: str | None = None) -> list[ModelDefinition]:
        if domain == "code":
            return [code_small, code_medium]
        return []

    assert isinstance(router.registry.list_models, Mock)
    router.registry.list_models.side_effect = list_models_side_effect

    model = router.route(context, user_id="user1")

    # Should pick the first one
    assert model.id == "code-small"
