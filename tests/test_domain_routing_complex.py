from unittest.mock import Mock

import pytest

from coreason_arbitrage.interfaces import BudgetClient
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
        provider="test",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.01,
        domain="medical",
    )
    medical_t3 = ModelDefinition(
        id="medical-t3",
        provider="test",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.10,
        cost_per_1k_output=0.10,
        domain="medical",
    )
    medical_unhealthy = ModelDefinition(
        id="medical-unhealthy",
        provider="test",
        tier=ModelTier.TIER_2_SMART,
        cost_per_1k_input=0.05,
        cost_per_1k_output=0.05,
        domain="medical",
        is_healthy=False,
    )

    # Generic models
    generic_t1 = ModelDefinition(
        id="generic-t1", provider="test", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0.01, cost_per_1k_output=0.01
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
def router(mock_registry_complex: Mock, mock_budget_client: Mock) -> Router:
    return Router(registry=mock_registry_complex, budget_client=mock_budget_client)


# --- Tests ---


def test_domain_model_selection_matches_complexity(router: Router) -> None:
    """
    Test that when multiple domain models exist, the one matching the complexity (Tier) is chosen.
    """
    # Case A: Low Complexity (0.1) -> Expects Tier 1
    # Should pick 'medical-t1'
    context_low = RoutingContext(complexity=0.1, domain="medical")
    model_low = router.route(context_low, user_id="user1")
    assert model_low.id == "medical-t1"
    assert model_low.tier == ModelTier.TIER_1_FAST

    # Case B: High Complexity (0.9) -> Expects Tier 3
    # Should pick 'medical-t3'
    context_high = RoutingContext(complexity=0.9, domain="medical")
    model_high = router.route(context_high, user_id="user1")
    assert model_high.id == "medical-t3"
    assert model_high.tier == ModelTier.TIER_3_REASONING


def test_domain_model_fallback_if_tier_mismatch(router: Router) -> None:
    """
    Test that if the requested Tier is not available in domain models,
    it falls back to ANY healthy domain model (preferring domain over generic).
    """
    # Complexity 0.5 -> Expects Tier 2
    # 'medical-unhealthy' is Tier 2 but unhealthy.
    # Healthy domain models: T1 and T3.
    # Should pick one of them (likely first in list or arbitrary, currently T1).
    context = RoutingContext(complexity=0.5, domain="medical")
    model = router.route(context, user_id="user1")

    assert model.domain == "medical"
    # Logic picks first healthy candidate if no tier match.
    # Our mock returns [t1, t3]. So it should match t1.
    assert model.id == "medical-t1"


def test_domain_model_respects_economy_mode(router: Router, mock_budget_client: Mock) -> None:
    """
    Test that Economy Mode downgrades the target tier, and then we look for a domain model matching that NEW tier.
    """
    mock_budget_client.get_remaining_budget_percentage.return_value = 0.05  # Broke

    # Complexity 0.5 (Tier 2). Economy Mode -> Downgrade to Tier 1.
    # We have Medical T1 and T3.
    # Should pick Medical T1.
    context = RoutingContext(complexity=0.5, domain="medical")
    model = router.route(context, user_id="user1")

    assert model.id == "medical-t1"
    assert model.tier == ModelTier.TIER_1_FAST


def test_multiple_domain_models_skips_unhealthy(router: Router) -> None:
    """
    Test that unhealthy domain models are skipped, even if they match the tier perfectly.
    """
    # Complexity 0.5 -> Tier 2.
    # We have 'medical-unhealthy' which is Tier 2.
    # It should be skipped.
    # Fallback to T1 or T3 (whichever is first healthy).
    context = RoutingContext(complexity=0.5, domain="medical")
    model = router.route(context, user_id="user1")

    assert model.id != "medical-unhealthy"
    assert model.domain == "medical"
