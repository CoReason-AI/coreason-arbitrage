from typing import List, Optional

from coreason_arbitrage.interfaces import AuditClient, BudgetClient, ModelFoundryClient
from coreason_arbitrage.models import ModelDefinition, ModelTier


class MockBudgetClient:
    def check_allowance(self, user_id: str) -> bool:
        return True

    def get_remaining_budget_percentage(self, user_id: str) -> float:
        return 0.5


class MockAuditClient:
    def log_transaction(
        self,
        user_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        pass


class MockModelFoundryClient:
    def list_custom_models(self, domain: Optional[str] = None) -> List[ModelDefinition]:
        return [
            ModelDefinition(
                id="custom/model-1",
                provider="custom",
                tier=ModelTier.TIER_2_SMART,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.02,
            )
        ]


def test_budget_client_protocol() -> None:
    # Verify that MockBudgetClient adheres to the BudgetClient protocol
    client: BudgetClient = MockBudgetClient()
    assert client.check_allowance("user1") is True
    assert client.get_remaining_budget_percentage("user1") == 0.5


def test_audit_client_protocol() -> None:
    # Verify that MockAuditClient adheres to the AuditClient protocol
    client: AuditClient = MockAuditClient()
    client.log_transaction("user1", "model1", 10, 20, 0.05)


def test_model_foundry_client_protocol() -> None:
    # Verify that MockModelFoundryClient adheres to the ModelFoundryClient protocol
    client: ModelFoundryClient = MockModelFoundryClient()
    models = client.list_custom_models()
    assert len(models) == 1
    assert models[0].id == "custom/model-1"
