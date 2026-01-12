import os
from unittest.mock import MagicMock, patch

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import BudgetClient


def reset_engine(engine: ArbitrageEngine) -> None:
    engine.registry.clear()
    if hasattr(engine, "load_balancer"):
        engine.load_balancer._failures.clear()
        engine.load_balancer._cooldown_until.clear()
    engine.budget_client = None
    engine.audit_client = None
    engine.foundry_client = None


def test_fail_open_default_fallback() -> None:
    """
    Test that SmartClient fails open to 'azure/gpt-4o' when router raises Exception,
    and logs a CRITICAL error.
    """
    engine = ArbitrageEngine()
    reset_engine(engine)

    # Configure with budget client to pass pre-flight check
    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    engine.configure(mock_budget, MagicMock(), MagicMock())

    # Get client
    client = engine.get_client()

    # Mock Router.route to raise Exception
    with patch.object(client.chat.completions.router, "route", side_effect=RuntimeError("Router Crashed")):
        with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
            with patch("coreason_arbitrage.smart_client.logger") as mock_logger:
                # Setup mock completion response
                mock_response = MagicMock()
                mock_completion.return_value = mock_response

                messages = [{"role": "user", "content": "Hello"}]
                response = client.chat.completions.create(messages=messages)

                # Assert fallback model was used
                mock_completion.assert_called_with(model="azure/gpt-4o", messages=messages)

                # Assert CRITICAL log
                critical_called = False
                for call in mock_logger.critical.call_args_list:
                    if "Fail-Open triggered" in str(call):
                        critical_called = True
                        break
                assert critical_called, "Expected CRITICAL log message 'Fail-Open triggered'"

                # Assert we got the response
                assert response == mock_response


def test_fail_open_custom_fallback_env_var() -> None:
    """
    Test that SmartClient fails open to ARBITRAGE_FALLBACK_MODEL when set.
    """
    engine = ArbitrageEngine()
    reset_engine(engine)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    engine.configure(mock_budget, MagicMock(), MagicMock())

    client = engine.get_client()

    custom_model = "aws/claude-3-sonnet"

    with patch.dict(os.environ, {"ARBITRAGE_FALLBACK_MODEL": custom_model}):
        with patch.object(client.chat.completions.router, "route", side_effect=RuntimeError("Router Crashed")):
            with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
                mock_completion.return_value = MagicMock()

                client.chat.completions.create(messages=[{"role": "user", "content": "Hello"}])

                mock_completion.assert_called_with(model=custom_model, messages=[{"role": "user", "content": "Hello"}])
