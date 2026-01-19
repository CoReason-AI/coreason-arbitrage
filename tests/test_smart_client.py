# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from unittest.mock import MagicMock, patch

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.smart_client import SmartClient


def reset_engine(engine: ArbitrageEngine) -> None:
    engine.registry.clear()
    if hasattr(engine, "load_balancer"):
        engine.load_balancer._failures.clear()
        engine.load_balancer._cooldown_until.clear()
    engine.budget_client = None
    engine.audit_client = None
    engine.foundry_client = None


def test_get_client_returns_smart_client() -> None:
    engine = ArbitrageEngine()
    client = engine.get_client()
    assert isinstance(client, SmartClient)
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client.chat.completions, "create")


def test_smart_client_gatekeeper_integration() -> None:
    engine = ArbitrageEngine()
    reset_engine(engine)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    mock_budget.get_remaining_budget_percentage.return_value = 0.5
    engine.configure(mock_budget, MagicMock(), MagicMock())

    # Register dummy model for Tier 3
    model = ModelDefinition(
        id="test-gpt4",
        provider="test",
        tier=ModelTier.TIER_3_REASONING,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
    )
    engine.registry.register_model(model)

    client = engine.get_client()

    messages = [{"role": "user", "content": "Analyze this protocol."}]

    with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
        mock_completion.return_value = MagicMock()
        client.chat.completions.create(messages=messages)
        mock_completion.assert_called_once()


def test_smart_client_gatekeeper_logic_called() -> None:
    # Use patching to verify Gatekeeper.classify is called
    with MagicMock():
        # We need to patch the Gatekeeper class used in smart_client.py
        from unittest.mock import patch

        with patch("coreason_arbitrage.smart_client.Gatekeeper") as MockGatekeeper:
            instance = MockGatekeeper.return_value
            # Return context that maps to TIER_1 to match our model
            instance.classify.return_value = MagicMock(complexity=0.1, domain=None)

            engine = ArbitrageEngine()
            reset_engine(engine)
            # Minimal config
            engine.configure(MagicMock(spec=BudgetClient), MagicMock(), MagicMock())
            # Register dummy model Tier 1
            model = ModelDefinition(
                id="test",
                provider="test",
                tier=ModelTier.TIER_1_FAST,
                cost_per_1k_input=0,
                cost_per_1k_output=0,
            )
            engine.registry.register_model(model)

            client = engine.get_client()

            with patch("coreason_arbitrage.smart_client.acompletion"):
                client.chat.completions.create(messages=[{"role": "user", "content": "test"}])

            instance.classify.assert_called_once_with("test")


def test_smart_client_missing_budget_client_warning() -> None:
    engine = ArbitrageEngine()
    reset_engine(engine)
    # Ensure budget_client is None
    assert engine.budget_client is None

    with patch("coreason_arbitrage.smart_client.logger") as mock_logger:
        engine.get_client()
        mock_logger.warning.assert_called_with("ArbitrageEngine not configured with BudgetClient. Router might fail.")


def test_smart_client_empty_prompt_warning() -> None:
    engine = ArbitrageEngine()
    reset_engine(engine)
    engine.configure(MagicMock(spec=BudgetClient), MagicMock(), MagicMock())
    engine.budget_client.check_allowance.return_value = True  # type: ignore

    # Register default Tier 1
    model = ModelDefinition(
        id="test",
        provider="test",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
    )
    engine.registry.register_model(model)

    client = engine.get_client()

    with patch("coreason_arbitrage.smart_client.logger") as mock_logger:
        with patch("coreason_arbitrage.smart_client.acompletion"):
            # Empty messages or no user message
            client.chat.completions.create(messages=[{"role": "system", "content": "sys"}])
            mock_logger.warning.assert_any_call(
                "No user message found in messages list. Using empty string for classification."
            )


def test_smart_client_audit_logging_failure() -> None:
    engine = ArbitrageEngine()
    reset_engine(engine)
    mock_audit = MagicMock()
    mock_audit.log_transaction.side_effect = Exception("Audit Log Error")

    engine.configure(MagicMock(spec=BudgetClient), mock_audit, MagicMock())
    engine.budget_client.check_allowance.return_value = True  # type: ignore

    model = ModelDefinition(
        id="test",
        provider="test",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
    )
    engine.registry.register_model(model)

    client = engine.get_client()

    with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
        mock_completion.return_value = MagicMock()

        with patch("coreason_arbitrage.smart_client.logger") as mock_logger:
            client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
            # Should log error but not fail request
            mock_logger.error.assert_called_with("Audit logging failed: Audit Log Error")


def test_smart_client_routing_failure_fails_open() -> None:
    """
    Test that when routing consistently fails (no healthy models),
    SmartClient fails open to the default fallback model.
    """
    engine = ArbitrageEngine()
    reset_engine(engine)

    engine.configure(MagicMock(spec=BudgetClient), MagicMock(), MagicMock())
    engine.budget_client.check_allowance.return_value = True  # type: ignore

    # Do NOT register any models -> routing will fail
    client = engine.get_client()

    with patch("coreason_arbitrage.smart_client.logger") as mock_logger:
        # Should NOT raise RuntimeError anymore, but return a response (mocked)
        with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
            mock_completion.return_value = MagicMock()

            client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

            # Verify fail open warning
            found_critical = False
            for call in mock_logger.critical.call_args_list:
                args, _ = call
                if "Fail-Open triggered" in str(args):
                    found_critical = True
                    break
            assert found_critical, "Expected 'Fail-Open triggered' log message"

            # Verify fallback model usage
            mock_completion.assert_called_with(model="azure/gpt-4o", messages=[{"role": "user", "content": "hi"}])


def test_smart_client_zero_retries_fail_open() -> None:
    """
    Test that if retries are zero, we immediately fail open.
    """
    engine = ArbitrageEngine()
    reset_engine(engine)
    engine.configure(MagicMock(spec=BudgetClient), MagicMock(), MagicMock())
    engine.budget_client.check_allowance.return_value = True  # type: ignore

    client = engine.get_client()

    with patch("coreason_arbitrage.smart_client.MAX_RETRIES", 0):
        with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
            mock_completion.return_value = MagicMock()

            client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

            mock_completion.assert_called_with(model="azure/gpt-4o", messages=[{"role": "user", "content": "hi"}])
