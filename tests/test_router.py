from unittest.mock import Mock

import pytest

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

    def list_models_side_effect(tier: ModelTier | None = None) -> list[ModelDefinition]:
        all_models = [t1, t2, t3]
        if tier:
            return [m for m in all_models if m.tier == tier]
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


# --- Tests ---


def test_route_tier_3_high_complexity(router: Router) -> None:
    """Test that high complexity routes to Tier 3."""
    context = RoutingContext(complexity=0.9, domain="general")
    model = router.route(context, user_id="user1")
    assert model.tier == ModelTier.TIER_3_REASONING
    assert model.id == "tier3-model"


def test_route_tier_3_safety_critical(router: Router) -> None:
    """Test that 'safety_critical' domain forces Tier 3 even with low complexity."""
    context = RoutingContext(complexity=0.1, domain="safety_critical")
    model = router.route(context, user_id="user1")
    assert model.tier == ModelTier.TIER_3_REASONING


def test_route_tier_2_mid_complexity(router: Router) -> None:
    """Test that mid complexity routes to Tier 2."""
    context = RoutingContext(complexity=0.5, domain="general")
    model = router.route(context, user_id="user1")
    assert model.tier == ModelTier.TIER_2_SMART


def test_route_tier_1_low_complexity(router: Router) -> None:
    """Test that low complexity routes to Tier 1."""
    context = RoutingContext(complexity=0.2, domain="general")
    model = router.route(context, user_id="user1")
    assert model.tier == ModelTier.TIER_1_FAST


def test_economy_mode_downgrade(router: Router, mock_budget_client: Mock) -> None:
    """Test that low budget (< 10%) downgrades Tier 2 to Tier 1."""
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.05  # 5%

    # Request that would normally be Tier 2
    context = RoutingContext(complexity=0.6, domain="general")
    model = router.route(context, user_id="broke_user")

    assert model.tier == ModelTier.TIER_1_FAST
    # Verify budget was checked
    mock_budget_client.get_remaining_budget_percentage.assert_called_with("broke_user")


def test_economy_mode_no_downgrade_for_tier_3(router: Router, mock_budget_client: Mock) -> None:
    """Test that low budget does NOT downgrade Tier 3 (Critical/Reasoning)."""
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.05

    context = RoutingContext(complexity=0.9, domain="general")
    model = router.route(context, user_id="broke_user")

    assert model.tier == ModelTier.TIER_3_REASONING


def test_budget_client_failure(router: Router, mock_budget_client: Mock) -> None:
    """Test that if BudgetClient fails, we proceed with baseline choice (Fail Open)."""
    mock_budget_client.get_remaining_budget_percentage.side_effect = Exception("DB Down")

    # Should be Tier 2 normally
    context = RoutingContext(complexity=0.6, domain="general")
    model = router.route(context, user_id="user1")

    assert model.tier == ModelTier.TIER_2_SMART


def test_no_models_available(router: Router, mock_registry: Mock) -> None:
    """Test that if registry has no models for the tier, an error is raised."""
    mock_registry.list_models.side_effect = None  # Clear fixture side_effect
    mock_registry.list_models.return_value = []  # Return empty list for any query

    context = RoutingContext(complexity=0.9, domain="general")

    with pytest.raises(RuntimeError, match="No healthy models available"):
        router.route(context, user_id="user1")


def test_unhealthy_models_skipped(router: Router, mock_registry: Mock) -> None:
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
    model = router.route(context, user_id="user1")

    assert model.id == "healthy_model"
