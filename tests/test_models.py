import pytest
from pydantic import ValidationError

from coreason_arbitrage.models import (
    ModelDefinition,
    ModelTier,
    RoutingContext,
    RoutingPolicy,
)


def test_model_tier_enum() -> None:
    assert ModelTier.TIER_1_FAST.value == "fast"
    assert ModelTier.TIER_2_SMART.value == "smart"
    assert ModelTier.TIER_3_REASONING.value == "reasoning"


def test_model_definition_valid() -> None:
    model = ModelDefinition(
        id="azure/gpt-4o",
        provider="azure",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.03,
        cost_per_1k_output=0.06,
    )
    assert model.id == "azure/gpt-4o"
    assert model.is_healthy is True


def test_model_definition_invalid() -> None:
    with pytest.raises(ValidationError):
        # We use a cast to trick MyPy into thinking this is valid code
        # so we can test the runtime validation.
        from typing import cast

        ModelDefinition(
            id="azure/gpt-4o",
            provider="azure",
            tier=cast(ModelTier, "invalid_tier"),
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.06,
        )


def test_routing_context_valid() -> None:
    context = RoutingContext(complexity=0.5, domain="medical")
    assert context.complexity == 0.5
    assert context.domain == "medical"


def test_routing_context_bounds() -> None:
    with pytest.raises(ValidationError):
        RoutingContext(complexity=1.1)  # > 1.0

    with pytest.raises(ValidationError):
        RoutingContext(complexity=-0.1)  # < 0.0


def test_routing_policy_valid() -> None:
    policy = RoutingPolicy(
        name="safety_critical",
        condition="complexity >= 0.8",
        models=["gpt-4o", "claude-3-opus"],
        fallback=["llama-3-70b-instruct"],
    )
    assert policy.name == "safety_critical"
    assert len(policy.models) == 2
