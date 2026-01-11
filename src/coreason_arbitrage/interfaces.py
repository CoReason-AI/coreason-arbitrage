from typing import List, Optional, Protocol, runtime_checkable

from coreason_arbitrage.models import ModelDefinition


@runtime_checkable
class BudgetClient(Protocol):
    """
    Protocol for interacting with the Budget service (coreason-budget).
    """

    def check_allowance(self, user_id: str) -> bool:
        """
        Checks if the user has enough budget to proceed with a request.
        """
        ...

    def get_remaining_budget_percentage(self, user_id: str) -> float:
        """
        Returns the user's remaining budget as a percentage (0.0 to 1.0).
        Used for "Economy Mode" decisions.
        """
        ...


@runtime_checkable
class AuditClient(Protocol):
    """
    Protocol for interacting with the Audit service (coreason-veritas).
    """

    def log_transaction(
        self,
        user_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """
        Logs a completed transaction for auditing and cost tracking.
        """
        ...


@runtime_checkable
class ModelFoundryClient(Protocol):
    """
    Protocol for interacting with the Model Foundry service (coreason-model-foundry).
    """

    def list_custom_models(self, domain: Optional[str] = None) -> List[ModelDefinition]:
        """
        Lists custom models available from the foundry, optionally filtered by domain.
        """
        ...
