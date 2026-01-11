from unittest.mock import MagicMock, patch

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.smart_client import SmartClient


def reset_engine(engine: ArbitrageEngine) -> None:
    # engine._models = {}  # _models is not in engine, it's in registry
    engine.registry.clear()
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

    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
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

            with patch("coreason_arbitrage.smart_client.completion"):
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
        with patch("coreason_arbitrage.smart_client.completion"):
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

    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        mock_completion.return_value = MagicMock()

        with patch("coreason_arbitrage.smart_client.logger") as mock_logger:
            client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
            # Should log error but not fail request
            mock_logger.error.assert_called_with("Audit logging failed: Audit Log Error")


def test_smart_client_routing_failure_logging() -> None:
    engine = ArbitrageEngine()
    reset_engine(engine)

    engine.configure(MagicMock(spec=BudgetClient), MagicMock(), MagicMock())
    engine.budget_client.check_allowance.return_value = True  # type: ignore

    # Do NOT register any models -> routing will fail
    client = engine.get_client()

    import pytest

    with patch("coreason_arbitrage.smart_client.logger") as mock_logger:
        with pytest.raises(RuntimeError, match="No healthy models"):
            client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

        # Verify logger.error was called with the routing exception
        args, _ = mock_logger.error.call_args_list[-1]
        assert "Routing failed:" in args[0]
