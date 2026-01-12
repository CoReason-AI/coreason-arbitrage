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
from typing import List
from unittest.mock import MagicMock

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import AuditClient, BudgetClient, ModelFoundryClient
from coreason_arbitrage.load_balancer import LoadBalancer
from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.registry import ModelRegistry


def reset_engine(engine: ArbitrageEngine) -> None:
    engine.registry.clear()
    if hasattr(engine, "load_balancer"):
        engine.load_balancer._failures.clear()
        engine.load_balancer._cooldown_until.clear()
    engine.budget_client = None
    engine.audit_client = None
    engine.foundry_client = None


def test_arbitrage_engine_singleton() -> None:
    engine1 = ArbitrageEngine()
    engine2 = ArbitrageEngine()
    assert engine1 is engine2
    assert engine1._initialized is True


def test_arbitrage_engine_initialization() -> None:
    engine = ArbitrageEngine()
    reset_engine(engine)  # Reset from previous tests

    assert isinstance(engine.load_balancer, LoadBalancer)
    assert isinstance(engine.registry, ModelRegistry)
    assert engine.budget_client is None
    assert engine.audit_client is None
    assert engine.foundry_client is None


def test_arbitrage_engine_configure() -> None:
    engine = ArbitrageEngine()
    reset_engine(engine)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_audit = MagicMock(spec=AuditClient)
    mock_foundry = MagicMock(spec=ModelFoundryClient)
    mock_foundry.list_custom_models.return_value = []  # Default return

    engine.configure(mock_budget, mock_audit, mock_foundry)

    assert engine.budget_client is mock_budget
    assert engine.audit_client is mock_audit
    assert engine.foundry_client is mock_foundry
    mock_foundry.list_custom_models.assert_called_once()


def test_arbitrage_engine_configure_pulls_models() -> None:
    """Test that configure() fetches models from foundry and registers them."""
    engine = ArbitrageEngine()
    reset_engine(engine)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_audit = MagicMock(spec=AuditClient)
    mock_foundry = MagicMock(spec=ModelFoundryClient)

    # Setup mock models
    model1 = ModelDefinition(
        id="custom/model-1",
        provider="custom",
        tier=ModelTier.TIER_2_SMART,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
        domain="medical",
    )
    model2 = ModelDefinition(
        id="custom/model-2",
        provider="custom",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0.05,
        cost_per_1k_output=0.10,
    )
    mock_foundry.list_custom_models.return_value = [model1, model2]

    engine.configure(mock_budget, mock_audit, mock_foundry)

    # Verify models are registered
    assert engine.registry.get_model("custom/model-1") == model1
    assert engine.registry.get_model("custom/model-2") == model2
    assert len(engine.registry._models) == 2


def test_arbitrage_engine_configure_foundry_failure() -> None:
    """Test that engine configuration succeeds even if pulling models fails."""
    engine = ArbitrageEngine()
    reset_engine(engine)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_audit = MagicMock(spec=AuditClient)
    mock_foundry = MagicMock(spec=ModelFoundryClient)

    # Simulate failure
    mock_foundry.list_custom_models.side_effect = RuntimeError("Connection failed")

    # Should not raise exception
    engine.configure(mock_budget, mock_audit, mock_foundry)

    assert engine.foundry_client is mock_foundry
    # Registry should be empty (or at least not have new models)
    assert len(engine.registry._models) == 0


def test_arbitrage_engine_thread_safety() -> None:
    """Test that the Singleton pattern holds up under concurrent access."""
    instances: List[ArbitrageEngine] = []

    def get_instance() -> None:
        instances.append(ArbitrageEngine())

    threads = [threading.Thread(target=get_instance) for _ in range(10)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify all instances are the same object
    first_instance = instances[0]
    for instance in instances[1:]:
        assert instance is first_instance

    # Verify initialized components are same
    assert instances[0].load_balancer is instances[1].load_balancer


def test_arbitrage_engine_reinitialization_idempotency() -> None:
    """Test that __init__ does not re-run initialization logic if called again."""
    engine = ArbitrageEngine()

    # Store original components
    lb = engine.load_balancer
    reg = engine.registry

    # Manually call __init__ again
    engine.__init__()  # type: ignore[misc]

    # Verify components haven't been overwritten
    assert engine.load_balancer is lb
    assert engine.registry is reg
    assert engine._initialized is True
