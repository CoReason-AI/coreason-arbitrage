# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

import random
import threading
import time

import pytest

from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.registry import ModelRegistry


@pytest.fixture
def registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.clear()
    return reg


def test_reference_behavior(registry: ModelRegistry) -> None:
    """
    Verify that modifying a retrieved model updates the state in the registry.
    This is critical for the 'Load Balancer' to update health status without re-registering.
    """
    model = ModelDefinition(
        id="test-ref",
        provider="test",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.1,
        cost_per_1k_output=0.1,
        is_healthy=True,
    )
    registry.register_model(model)

    # Retrieve and modify
    retrieved = registry.get_model("test-ref")
    assert retrieved is not None
    assert retrieved.is_healthy is True

    retrieved.is_healthy = False

    # Retrieve again to verify persistence
    retrieved_again = registry.get_model("test-ref")
    assert retrieved_again is not None
    assert retrieved_again.is_healthy is False

    # Verify it is indeed the same object reference
    assert retrieved is retrieved_again


def test_data_overwrite_integrity(registry: ModelRegistry) -> None:
    """
    Verify that re-registering a model with the same ID but different data
    correctly updates all fields.
    """
    # Initial registration
    model_v1 = ModelDefinition(
        id="test-overwrite",
        provider="azure",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.01,
        is_healthy=True,
    )
    registry.register_model(model_v1)

    # Verify v1
    retrieved = registry.get_model("test-overwrite")
    assert retrieved is not None
    assert retrieved.provider == "azure"
    assert retrieved.tier == ModelTier.TIER_1_FAST

    # Overwrite with v2 (different provider, different tier)
    model_v2 = ModelDefinition(
        id="test-overwrite",
        provider="aws",
        tier=ModelTier.TIER_2_SMART,
        cost_per_1k_input=0.02,
        cost_per_1k_output=0.02,
        is_healthy=False,
    )
    registry.register_model(model_v2)

    # Verify v2
    retrieved_v2 = registry.get_model("test-overwrite")
    assert retrieved_v2 is not None
    assert retrieved_v2.provider == "aws"
    assert retrieved_v2.tier == ModelTier.TIER_2_SMART
    assert retrieved_v2.is_healthy is False


def test_complex_concurrency_mixed_load(registry: ModelRegistry) -> None:
    """
    Simulate a high-concurrency environment with:
    - Writers: Registering new models.
    - Updaters: Changing health status of existing models.
    - Readers: Listing and retrieving models.
    """
    # Pre-populate some models
    base_models = [f"model-{i}" for i in range(10)]
    for mid in base_models:
        registry.register_model(
            ModelDefinition(
                id=mid, provider="setup", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0.1, cost_per_1k_output=0.1
            )
        )

    stop_event = threading.Event()
    errors: list[Exception] = []

    def writer_task() -> None:
        try:
            for i in range(10, 50):
                if stop_event.is_set():
                    break
                model = ModelDefinition(
                    id=f"model-{i}",
                    provider="writer",
                    tier=ModelTier.TIER_1_FAST,
                    cost_per_1k_input=0.1,
                    cost_per_1k_output=0.1,
                )
                registry.register_model(model)
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    def updater_task() -> None:
        try:
            while not stop_event.is_set():
                # Pick a random model (0-9 are guaranteed to exist initially)
                mid = f"model-{random.randint(0, 9)}"
                model = registry.get_model(mid)
                if model:
                    # Toggle health
                    model.is_healthy = not model.is_healthy
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    def reader_task() -> None:
        try:
            while not stop_event.is_set():
                # List all models
                models = registry.list_models()
                assert len(models) >= 10

                # Get specific model
                mid = f"model-{random.randint(0, 9)}"
                m = registry.get_model(mid)
                assert m is not None
                assert m.id == mid
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=writer_task),
        threading.Thread(target=updater_task),
        threading.Thread(target=reader_task),
        threading.Thread(target=reader_task),
    ]

    for t in threads:
        t.start()

    time.sleep(0.5)  # Run load for 0.5 seconds
    stop_event.set()

    for t in threads:
        t.join()

    assert not errors, f"Encountered errors in threads: {errors}"

    # Final consistency check
    final_count = len(registry.list_models())
    assert final_count >= 10


def test_list_models_domain_filter_advanced(registry: ModelRegistry) -> None:
    """
    Test edge cases for domain filtering in ModelRegistry.
    Specifically targeting the list comprehension filter logic.
    """
    # 1. Model with domain
    m_domain = ModelDefinition(
        id="m1", provider="p", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0, cost_per_1k_output=0, domain="Finance"
    )
    # 2. Model without domain (None)
    m_no_domain = ModelDefinition(
        id="m2", provider="p", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0, cost_per_1k_output=0, domain=None
    )
    # 3. Model with different domain
    m_other = ModelDefinition(
        id="m3", provider="p", tier=ModelTier.TIER_1_FAST, cost_per_1k_input=0, cost_per_1k_output=0, domain="Medical"
    )

    registry.register_model(m_domain)
    registry.register_model(m_no_domain)
    registry.register_model(m_other)

    # Search for "finance" (case insensitive)
    # Should find m_domain
    # Should skip m_no_domain (m.domain is None)
    # Should skip m_other (domain mismatch)
    results = registry.list_models(domain="finance")

    assert len(results) == 1
    assert results[0].id == "m1"

    # Verify strictness: searching for something else returns empty
    assert len(registry.list_models(domain="Legal")) == 0
