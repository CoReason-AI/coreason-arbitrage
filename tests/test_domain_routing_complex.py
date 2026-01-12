from unittest.mock import Mock

import pytest

from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.load_balancer import LoadBalancer
from coreason_arbitrage.models import ModelDefinition, ModelTier, RoutingContext
from coreason_arbitrage.registry import ModelRegistry
from coreason_arbitrage.router import Router

# --- Fixtures ---


@pytest.fixture
def mock_registry_complex() -> Mock:
    registry = Mock(spec=ModelRegistry)

    # Setup a scenario with multiple domain models
    # Domain: "medical"
    # - medical-t1: Tier 1, Healthy
    # - medical-t3: Tier 3, Healthy
    # - medical-unhealthy: Tier 2, Unhealthy

    medical_t1 = ModelDefinition(
        id="medical-t1",
        provider="provider-a",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.01,
        domain="medical",
    )
    medical_t3 = ModelDefinition(
        id="medical-t3",
        provider="provider-a",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.10,
        cost_per_1k_output=0.10,
        domain="medical",
    )
    medical_unhealthy = ModelDefinition(
        id="medical-unhealthy",
        provider="provider-b",
        tier=ModelTier.TIER_2_SMART,
        cost_per_1k_input=0.05,
        cost_per_1k_output=0.05,
        domain="medical",
        is_healthy=False,
    )

    # Generic models
    generic_t1 = ModelDefinition(
        id="generic-t1",
        provider="provider-c",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.01,
    )

    all_models = [medical_t1, medical_t3, medical_unhealthy, generic_t1]

    def list_models_side_effect(tier: ModelTier | None = None, domain: str | None = None) -> list[ModelDefinition]:
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
def mock_load_balancer() -> Mock:
    lb = Mock(spec=LoadBalancer)
    # Default all healthy
    lb.is_provider_healthy.return_value = True
    return lb


@pytest.fixture
def router(mock_registry_complex: Mock, mock_budget_client: Mock) -> Router:
    return Router(registry=mock_registry_complex, budget_client=mock_budget_client)


# --- Tests ---


def test_domain_model_selection_matches_complexity(router: Router) -> None:
    """
    Test that when multiple domain models exist, the one matching the complexity (Tier) is chosen.
    """
    context_low = RoutingContext(complexity=0.1, domain="medical")
    model_low = router.route(context_low, user_id="user1")
    assert model_low.id == "medical-t1"
    assert model_low.tier == ModelTier.TIER_1_FAST

    context_high = RoutingContext(complexity=0.9, domain="medical")
    model_high = router.route(context_high, user_id="user1")
    assert model_high.id == "medical-t3"
    assert model_high.tier == ModelTier.TIER_3_REASONING


def test_domain_model_fallback_if_tier_mismatch(router: Router) -> None:
    """
    Test that if the requested Tier is not available in domain models,
    it falls back to ANY healthy domain model (preferring domain over generic).
    """
    context = RoutingContext(complexity=0.5, domain="medical")
    model = router.route(context, user_id="user1")

    assert model.domain == "medical"
    assert model.id == "medical-t1"


def test_domain_model_respects_economy_mode(router: Router, mock_budget_client: Mock) -> None:
    """
    Test that Economy Mode downgrades the target tier, and then we look for a domain model matching that NEW tier.
    """
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.05  # Broke

    context = RoutingContext(complexity=0.5, domain="medical")
    model = router.route(context, user_id="user1")

    assert model.id == "medical-t1"
    assert model.tier == ModelTier.TIER_1_FAST


def test_multiple_domain_models_skips_unhealthy(router: Router) -> None:
    """
    Test that unhealthy domain models are skipped, even if they match the tier perfectly.
    """
    context = RoutingContext(complexity=0.5, domain="medical")
    model = router.route(context, user_id="user1")

    assert model.id != "medical-unhealthy"
    assert model.domain == "medical"


def test_domain_model_filtered_by_load_balancer(
    mock_registry_complex: Mock, mock_budget_client: Mock, mock_load_balancer: Mock
) -> None:
    """
    Test that domain models are filtered by the Load Balancer if it is present.
    """
    # Create router WITH load balancer
    router_with_lb = Router(
        registry=mock_registry_complex, budget_client=mock_budget_client, load_balancer=mock_load_balancer
    )

    # Setup: 'medical-t1' provider is UNHEALTHY via LoadBalancer
    # 'medical-t3' provider is HEALTHY

    # medical-t1 uses 'provider-a'
    # medical-t3 uses 'provider-a'
    # Wait, let's make them use different providers or mock side effect

    def is_provider_healthy(provider: str) -> bool:
        if provider == "provider-a":
            return False
        return True

    mock_load_balancer.is_provider_healthy.side_effect = is_provider_healthy

    # If we request complexity 0.1, normally we get medical-t1 (Tier 1).
    # But provider-a is unhealthy.
    # So we should skip it.

    # Wait, both my mock models use 'provider-a' in the fixture above.
    # I should update fixture or creating new models here?
    # I'll rely on the fixture but update the test logic expectation.
    # If both t1 and t3 use provider-a and it's down, we have NO domain models.
    # Then it should fallback to generic.

    # Let's change the fixture slightly to have distinct providers?
    # Or just Mock the registry again locally.

    # Let's verify fallback to generic if all domain providers are down.
    context = RoutingContext(complexity=0.1, domain="medical")

    # Expectation: Falls back to generic-t1 (if healthy)
    # generic-t1 uses provider-c, which is healthy

    model = router_with_lb.route(context, user_id="user1")

    assert model.id == "generic-t1"
    assert model.domain is None
