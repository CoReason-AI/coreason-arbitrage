import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import AuditClient, BudgetClient


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


def test_fail_open_cascading_failure() -> None:
    """
    Test scenario where both the Router crashes AND the Fallback model execution fails.
    Should raise the ORIGINAL exception (Router Crashed) with context.
    """
    engine = ArbitrageEngine()
    reset_engine(engine)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    engine.configure(mock_budget, MagicMock(), MagicMock())

    client = engine.get_client()

    with patch.object(client.chat.completions.router, "route", side_effect=RuntimeError("Router Crashed")):
        with patch("coreason_arbitrage.smart_client.completion", side_effect=Exception("Fallback Failed")):
            with patch("coreason_arbitrage.smart_client.logger") as mock_logger:
                # We expect the ORIGINAL exception (Router Crashed) to be raised
                with pytest.raises(RuntimeError, match="Router Crashed"):
                    client.chat.completions.create(messages=[{"role": "user", "content": "Hello"}])

                # Verify critical logs
                # We expect at least one critical log (Fail-Open triggered)
                assert mock_logger.critical.call_count >= 1
                assert "Fail-Open triggered" in str(mock_logger.critical.call_args_list[0])


def test_fail_open_audit_failure() -> None:
    """
    Test scenario where Fail-Open succeeds, but Audit logging fails.
    The response should still be returned (Fail Open for Audit too).
    """
    engine = ArbitrageEngine()
    reset_engine(engine)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True

    mock_audit = MagicMock(spec=AuditClient)
    mock_audit.log_transaction.side_effect = Exception("Audit DB Down")

    engine.configure(mock_budget, mock_audit, MagicMock())

    client = engine.get_client()

    with patch.object(client.chat.completions.router, "route", side_effect=RuntimeError("Router Crashed")):
        with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
            with patch("coreason_arbitrage.smart_client.logger") as mock_logger:
                mock_response = MagicMock()
                mock_completion.return_value = mock_response

                response = client.chat.completions.create(messages=[{"role": "user", "content": "Hello"}])

                assert response == mock_response

                # Verify Error log for Audit
                mock_logger.error.assert_called_with("Audit logging failed during fail-open: Audit DB Down")


def test_fail_open_invalid_env_var() -> None:
    """
    Test fail-open with an invalid (empty) fallback model ID.
    Pydantic validation should fail when creating ModelDefinition.
    """
    engine = ArbitrageEngine()
    reset_engine(engine)

    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    engine.configure(mock_budget, MagicMock(), MagicMock())

    client = engine.get_client()

    # Set invalid model (empty string)
    with patch.dict(os.environ, {"ARBITRAGE_FALLBACK_MODEL": ""}):
        with patch.object(client.chat.completions.router, "route", side_effect=RuntimeError("Router Crashed")):
            # Expect Pydantic ValidationError because ModelDefinition requires min_length=1
            with pytest.raises(ValidationError):
                client.chat.completions.create(messages=[{"role": "user", "content": "Hello"}])


def test_fail_open_immediate_failure_with_zero_retries() -> None:
    """
    Test fail-open execution when MAX_RETRIES is 0 and fallback ALSO fails.
    This triggers the `raise e from None` path because last_exception is None.
    """
    engine = ArbitrageEngine()
    reset_engine(engine)
    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    engine.configure(mock_budget, MagicMock(), MagicMock())
    client = engine.get_client()

    with patch("coreason_arbitrage.smart_client.MAX_RETRIES", 0):
        with patch("coreason_arbitrage.smart_client.completion", side_effect=Exception("Fallback Error")):
            with pytest.raises(Exception, match="Fallback Error"):
                client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
