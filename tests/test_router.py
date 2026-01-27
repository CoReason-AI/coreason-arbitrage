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

    # Define standard models for each tier
    t1 = ModelDefinition(
        id="tier1-model", provider="test", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0.01, cost_per_1k_output=0.01
    )
    t2 = ModelDefinition(
        id="tier2-model", provider="test", tier=ModelTier.TIER_2_SMART, cost_per_1k_input=0.05, cost_per_1k_output=0.05
    )
    t3 = ModelDefinition(
        id="tier3-model",
        provider="test",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.10,
        cost_per_1k_output=0.10,
    )

    def list_models_side_effect(tier: ModelTier | None = None, domain: str | None = None) -> list[ModelDefinition]:
        all_models = [t1, t2, t3]
        if tier:
            return [m for m in all_models if m.tier == tier]
        # For these existing tests, we don't really simulate domain matching
        # unless specifically testing it, so we can ignore 'domain' arg or return [] if it's set
        # to ensure fallback logic is triggered (since these tests expect fallback/tier logic).
        if domain:
            # If domain is requested, and we have no specific domain models in this fixture, return empty
            return []
        return all_models

    registry.list_models.side_effect = list_models_side_effect
    return registry


@pytest.fixture
def mock_budget_client() -> Mock:
    client = Mock(spec=BudgetClient)
    # Default: plenty of budget
    client.get_remaining_budget_percentage.return_value = 1.0
    return client


@pytest.fixture
def router(mock_registry: Mock, mock_budget_client: Mock) -> Router:
    return Router(registry=mock_registry, budget_client=mock_budget_client)


@pytest.fixture
def user_context() -> UserContext:
    # Assuming UserContext has defaults or minimal requirements
    # Since I don't know the exact schema of UserContext from coreason-identity (external lib),
    # I'll rely on the prompt's implication that it has user_id and groups.
    # If it's a Pydantic model, I should instantiate it correctly.
    # If I can't instantiate it because of missing fields, I'll use a Mock that looks like it.
    # But for now let's try to instantiate it assuming it takes kwargs or has defaults.
    # Actually, better to use Mock(spec=UserContext) to avoid schema validation issues if fields are missing
    # But since the code uses user_context.user_id, a Mock is safer if I don't know required fields.
    # However, type hinting expects UserContext.
    # Let's try to create a real one if possible, or Mock.
    # Given I can't see the UserContext definition, I'll use a Mock for the fixture to be safe,
    # but I'll make sure it passes isinstance check if needed (it's not checked with isinstance
    # in code, just type hint).
    uc = Mock(spec=UserContext)
    uc.user_id = "user1"
    uc.groups = ["users"]
    return uc


# --- Standard Tests ---


def test_route_tier_3_high_complexity(router: Router, user_context: UserContext) -> None:
    """Test that high complexity routes to Tier 3."""
    context = RoutingContext(complexity=0.9, domain="general")
    model = router.route(context, user_context=user_context)
    assert model.tier == ModelTier.TIER_3_REASONING
    assert model.id == "tier3-model"


def test_route_tier_3_safety_critical(router: Router, user_context: UserContext) -> None:
    """Test that 'safety_critical' domain forces Tier 3 even with low complexity."""
    context = RoutingContext(complexity=0.1, domain="safety_critical")
    model = router.route(context, user_context=user_context)
    assert model.tier == ModelTier.TIER_3_REASONING


def test_route_tier_2_mid_complexity(router: Router, user_context: UserContext) -> None:
    """Test that mid complexity routes to Tier 2."""
    context = RoutingContext(complexity=0.5, domain="general")
    model = router.route(context, user_context=user_context)
    assert model.tier == ModelTier.TIER_2_SMART


def test_route_tier_1_low_complexity(router: Router, user_context: UserContext) -> None:
    """Test that low complexity routes to Tier 1."""
    context = RoutingContext(complexity=0.2, domain="general")
    model = router.route(context, user_context=user_context)
    assert model.tier == ModelTier.TIER_1_FAST


def test_economy_mode_downgrade(router: Router, mock_budget_client: Mock) -> None:
    """Test that low budget (< 10%) downgrades Tier 2 to Tier 1."""
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.05  # 5%

    uc = Mock(spec=UserContext)
    uc.user_id = "broke_user"
    uc.groups = ["users"]

    # Request that would normally be Tier 2
    context = RoutingContext(complexity=0.6, domain="general")
    model = router.route(context, user_context=uc)

    assert model.tier == ModelTier.TIER_1_FAST
    # Verify budget was checked
    mock_budget_client.get_remaining_budget_percentage.assert_called_with(uc)


def test_economy_mode_no_downgrade_for_tier_3(router: Router, mock_budget_client: Mock) -> None:
    """Test that low budget does NOT downgrade Tier 3 (Critical/Reasoning)."""
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.05

    uc = Mock(spec=UserContext)
    uc.user_id = "broke_user"
    uc.groups = ["users"]

    context = RoutingContext(complexity=0.9, domain="general")
    model = router.route(context, user_context=uc)

    assert model.tier == ModelTier.TIER_3_REASONING


def test_budget_client_failure(router: Router, mock_budget_client: Mock, user_context: UserContext) -> None:
    """Test that if BudgetClient fails, we proceed with baseline choice (Fail Open)."""
    mock_budget_client.get_remaining_budget_percentage.side_effect = Exception("DB Down")

    # Should be Tier 2 normally
    context = RoutingContext(complexity=0.6, domain="general")
    model = router.route(context, user_context=user_context)

    assert model.tier == ModelTier.TIER_2_SMART


def test_no_models_available(router: Router, mock_registry: Mock, user_context: UserContext) -> None:
    """Test that if registry has no models for the tier, an error is raised."""
    mock_registry.list_models.side_effect = None  # Clear fixture side_effect
    mock_registry.list_models.return_value = []  # Return empty list for any query

    context = RoutingContext(complexity=0.9, domain="general")

    with pytest.raises(RuntimeError, match="No healthy models available"):
        router.route(context, user_context=user_context)


def test_unhealthy_models_skipped(router: Router, mock_registry: Mock, user_context: UserContext) -> None:
    """Test that unhealthy models are skipped."""
    # Create an unhealthy Tier 1 model
    unhealthy_model = ModelDefinition(
        id="sick_model",
        provider="test",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.01,
        is_healthy=False,
    )
    healthy_model = ModelDefinition(
        id="healthy_model",
        provider="test",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.01,
        is_healthy=True,
    )

    # Registry returns both (unhealthy first to test skipping)
    mock_registry.list_models.side_effect = None  # Clear fixture side_effect
    mock_registry.list_models.return_value = [unhealthy_model, healthy_model]

    context = RoutingContext(complexity=0.1, domain="general")
    model = router.route(context, user_context=user_context)

    assert model.id == "healthy_model"


# --- Edge Case & Complex Scenario Tests ---


def test_complexity_boundaries(router: Router, user_context: UserContext) -> None:
    """Test exact boundary values for complexity."""
    # 0.3999 -> Tier 1
    context = RoutingContext(complexity=0.3999, domain="general")
    assert router.route(context, user_context).tier == ModelTier.TIER_1_FAST

    # 0.4 -> Tier 2
    context = RoutingContext(complexity=0.4, domain="general")
    assert router.route(context, user_context).tier == ModelTier.TIER_2_SMART

    # 0.7999 -> Tier 2
    context = RoutingContext(complexity=0.7999, domain="general")
    assert router.route(context, user_context).tier == ModelTier.TIER_2_SMART

    # 0.8 -> Tier 3
    context = RoutingContext(complexity=0.8, domain="general")
    assert router.route(context, user_context).tier == ModelTier.TIER_3_REASONING


def test_budget_boundaries(router: Router, mock_budget_client: Mock, user_context: UserContext) -> None:
    """Test exact boundary values for economy mode budget."""
    # 0.10 (10%) -> No downgrade
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.10
    context = RoutingContext(complexity=0.5, domain="general")  # Tier 2 target
    assert router.route(context, user_context).tier == ModelTier.TIER_2_SMART

    # 0.0999 (9.99%) -> Downgrade
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.0999
    context = RoutingContext(complexity=0.5, domain="general")  # Tier 2 target
    assert router.route(context, user_context).tier == ModelTier.TIER_1_FAST


def test_domain_case_insensitivity(router: Router, user_context: UserContext) -> None:
    """Test that domain check is case insensitive."""
    # "Safety_Critical" -> Should trigger Tier 3 (High)
    context = RoutingContext(complexity=0.1, domain="Safety_Critical")
    model = router.route(context, user_context)
    assert model.tier == ModelTier.TIER_3_REASONING

    # "SAFETY_CRITICAL" -> Should trigger Tier 3 (High)
    context = RoutingContext(complexity=0.1, domain="SAFETY_CRITICAL")
    model = router.route(context, user_context)
    assert model.tier == ModelTier.TIER_3_REASONING


def test_domain_null_or_empty(router: Router, user_context: UserContext) -> None:
    """Test None or Empty string domains don't crash."""
    # None
    context = RoutingContext(complexity=0.2, domain=None)
    assert router.route(context, user_context).tier == ModelTier.TIER_1_FAST

    # Empty
    context = RoutingContext(complexity=0.2, domain="")
    assert router.route(context, user_context).tier == ModelTier.TIER_1_FAST


def test_economy_downgrade_dead_end(router: Router, mock_budget_client: Mock, mock_registry: Mock) -> None:
    """
    Test scenario:
    - User wants Tier 2.
    - Budget is low -> Downgrades to Tier 1.
    - Tier 1 has NO models.
    - Result: Should raise RuntimeError (Fail Open/Safe).
    """
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.05

    uc = Mock(spec=UserContext)
    uc.user_id = "u1"
    uc.groups = []

    # Mock registry: Tier 2 exists, Tier 1 is empty
    t2 = ModelDefinition(
        id="tier2-model", provider="test", tier=ModelTier.TIER_2_SMART, cost_per_1k_input=0.05, cost_per_1k_output=0.05
    )

    def list_models_dead_end(tier: ModelTier | None = None, domain: str | None = None) -> list[ModelDefinition]:
        if domain:
            return []
        if tier == ModelTier.TIER_2_SMART:
            return [t2]
        if tier == ModelTier.TIER_1_FAST:
            return []  # No cheap models!
        return []

    mock_registry.list_models.side_effect = list_models_dead_end

    context = RoutingContext(complexity=0.5, domain="general")  # Target Tier 2

    # Expect error because it tries to find Tier 1 and fails
    with pytest.raises(RuntimeError, match="No healthy models available for Tier"):
        router.route(context, user_context=uc)


# --- New Tests for Identity & VIP ---


def test_missing_user_context_enforces_economy(router: Router, mock_budget_client: Mock) -> None:
    """Test that missing user_context forces Tier 1 (Economy Mode) and skips budget check."""
    # Would be Tier 2 based on complexity
    context = RoutingContext(complexity=0.5, domain="general")

    # Call without user_context
    model = router.route(context, user_context=None)

    assert model.tier == ModelTier.TIER_1_FAST
    # Budget check should NOT be called
    mock_budget_client.get_remaining_budget_percentage.assert_not_called()


def test_vip_user_skips_economy_check(router: Router, mock_budget_client: Mock) -> None:
    """Test that VIP users (Executives) are not downgraded even with low budget."""
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.05  # 5% budget

    uc = Mock(spec=UserContext)
    uc.user_id = "vip_user"
    uc.groups = [
        "staff",
        "Executives",
    ]  # Case sensitive check in code was implemented as insensitive? I should check code.
    # My implementation: any(group.lower() == "executives" for group in user_context.groups)

    # Would be Tier 2
    context = RoutingContext(complexity=0.5, domain="general")

    model = router.route(context, user_context=uc)

    # Should stay Tier 2 because VIP
    assert model.tier == ModelTier.TIER_2_SMART
    # Budget check should be skipped
    mock_budget_client.get_remaining_budget_percentage.assert_not_called()


def test_vip_user_case_insensitive(router: Router, mock_budget_client: Mock) -> None:
    """Test that VIP check is case insensitive."""
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.05

    uc = Mock(spec=UserContext)
    uc.user_id = "vip_user"
    uc.groups = ["executives"]  # Lowercase

    context = RoutingContext(complexity=0.5, domain="general")
    model = router.route(context, user_context=uc)

    assert model.tier == ModelTier.TIER_2_SMART
