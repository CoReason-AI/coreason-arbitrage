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

import pytest

from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.registry import ModelRegistry


@pytest.fixture
def registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.clear()
    return reg


@pytest.fixture
def sample_model_fast() -> ModelDefinition:
    return ModelDefinition(
        id="model-fast",
        provider="provider-a",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.01,
    )


@pytest.fixture
def sample_model_smart() -> ModelDefinition:
    return ModelDefinition(
        id="model-smart",
        provider="provider-b",
        tier=ModelTier.TIER_2_SMART,
        cost_per_1k_input=0.02,
        cost_per_1k_output=0.02,
    )


def test_singleton_behavior(registry: ModelRegistry) -> None:
    reg1 = ModelRegistry()
    reg2 = ModelRegistry()
    assert reg1 is reg2
    assert reg1._models is reg2._models


def test_register_and_get_model(registry: ModelRegistry, sample_model_fast: ModelDefinition) -> None:
    registry.register_model(sample_model_fast)
    retrieved = registry.get_model("model-fast")
    assert retrieved == sample_model_fast


def test_list_models(
    registry: ModelRegistry, sample_model_fast: ModelDefinition, sample_model_smart: ModelDefinition
) -> None:
    registry.register_model(sample_model_fast)
    registry.register_model(sample_model_smart)

    all_models = registry.list_models()
    assert len(all_models) == 2

    fast_models = registry.list_models(tier=ModelTier.TIER_1_FAST)
    assert len(fast_models) == 1
    assert fast_models[0].id == "model-fast"

    smart_models = registry.list_models(tier=ModelTier.TIER_2_SMART)
    assert len(smart_models) == 1
    assert smart_models[0].id == "model-smart"


def test_update_existing_model(registry: ModelRegistry, sample_model_fast: ModelDefinition) -> None:
    registry.register_model(sample_model_fast)

    # Update health status
    updated_model = sample_model_fast.model_copy()
    updated_model.is_healthy = False

    registry.register_model(updated_model)

    retrieved = registry.get_model("model-fast")
    assert retrieved is not None
    assert retrieved.is_healthy is False


def test_thread_safety(registry: ModelRegistry) -> None:
    # Basic check to ensure no exceptions during concurrent access

    def register_many() -> None:
        for i in range(100):
            model = ModelDefinition(
                id=f"model-{i}",
                provider="test",
                tier=ModelTier.TIER_1_FAST,
                cost_per_1k_input=0.1,
                cost_per_1k_output=0.1,
            )
            registry.register_model(model)

    threads = [threading.Thread(target=register_many) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(registry.list_models()) == 100


def test_singleton_init_race() -> None:
    # Force reset
    ModelRegistry._instance = None

    inst = ModelRegistry()  # This runs init once.
    inst._initialized = False  # Reset flag manually to simulate race start

    def run_init() -> None:
        inst.__init__()  # type: ignore[misc]

    threads = [threading.Thread(target=run_init) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert inst._initialized is True
