from unittest.mock import MagicMock, patch

import pytest

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.interfaces import AuditClient, BudgetClient
from coreason_arbitrage.models import ModelDefinition, ModelTier


@pytest.fixture
def configured_engine() -> ArbitrageEngine:
    engine = ArbitrageEngine()
    engine.registry.clear()

    # Check if load_balancer is initialized before accessing it.
    # It should be initialized by engine.__init__
    if hasattr(engine, "load_balancer"):
        engine.load_balancer._failures.clear()
        engine.load_balancer._cooldown_until.clear()

    mock_budget = MagicMock(spec=BudgetClient)
    mock_budget.check_allowance.return_value = True
    mock_budget.get_remaining_budget_percentage.return_value = 0.5

    mock_audit = MagicMock(spec=AuditClient)
    mock_foundry = MagicMock()

    engine.configure(mock_budget, mock_audit, mock_foundry)

    model = ModelDefinition(
        id="test-model",
        provider="test-provider",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
    )
    engine.registry.register_model(model)

    return engine


def test_smart_client_execution_success(configured_engine: ArbitrageEngine) -> None:
    client = configured_engine.get_client()
    messages = [{"role": "user", "content": "hello"}]

    # Mock litellm.completion
    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_completion.return_value = mock_response

        response = client.chat.completions.create(messages=messages, user="test_user")

        assert response is mock_response
        mock_completion.assert_called_once()

        # Verify LB success recorded
        assert configured_engine.load_balancer.is_provider_healthy("test-provider")


def test_smart_client_audit_logging(configured_engine: ArbitrageEngine) -> None:
    client = configured_engine.get_client()
    messages = [{"role": "user", "content": "hello"}]

    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 1000
        mock_completion.return_value = mock_response

        client.chat.completions.create(messages=messages, user="test_user")

        # Verify audit log
        # Cost = 1 * 0.01 + 1 * 0.02 = 0.03
        assert configured_engine.audit_client is not None
        configured_engine.audit_client.log_transaction.assert_called_once()  # type: ignore
        args = configured_engine.audit_client.log_transaction.call_args[1]  # type: ignore
        assert args["user_id"] == "test_user"
        assert args["model_id"] == "test-model"
        assert args["input_tokens"] == 1000
        assert args["output_tokens"] == 1000
        assert args["cost"] == pytest.approx(0.03)


def test_smart_client_execution_failure_updates_lb(
    configured_engine: ArbitrageEngine,
) -> None:
    client = configured_engine.get_client()
    messages = [{"role": "user", "content": "hello"}]

    with patch("coreason_arbitrage.smart_client.completion") as mock_completion:
        mock_completion.side_effect = Exception("API Error")

        # LB should be updated
        with patch.object(configured_engine.load_balancer, "record_failure") as mock_record_failure:
            with pytest.raises(Exception, match="API Error"):
                client.chat.completions.create(messages=messages, user="test_user")

            mock_record_failure.assert_called_with("test-provider")
