# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from typing import List, Optional, Protocol, runtime_checkable

from coreason_identity.models import UserContext

from coreason_arbitrage.models import ModelDefinition


@runtime_checkable
class BudgetClient(Protocol):
    """Protocol for interacting with the Budget service (coreason-budget)."""

    def check_allowance(self, user_id: str) -> bool:
        """Checks if the user has enough budget to proceed with a request.

        Args:
            user_id: The ID of the user.

        Returns:
            True if the user has sufficient funds, False otherwise.
        """
        ...

    def get_remaining_budget_percentage(self, user_context: UserContext) -> float:
        """Returns the user's remaining budget as a percentage (0.0 to 1.0).

        Used for "Economy Mode" decisions.

        Args:
            user_context: The verified user context.

        Returns:
            A float representing the remaining budget percentage (e.g., 0.15 for 15%).
        """
        ...

    def deduct_funds(self, user_id: str, amount: float) -> None:
        """Deducts the specified amount from the user's budget.

        Args:
            user_id: The ID of the user.
            amount: The amount to deduct (typically in USD).
        """
        ...


@runtime_checkable
class AuditClient(Protocol):
    """Protocol for interacting with the Audit service (coreason-veritas)."""

    def log_transaction(
        self,
        user_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Logs a completed transaction for auditing and cost tracking.

        Args:
            user_id: The ID of the user who initiated the request.
            model_id: The ID of the model used.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens consumed.
            cost: Total cost of the transaction.
        """
        ...


@runtime_checkable
class ModelFoundryClient(Protocol):
    """Protocol for interacting with the Model Foundry service (coreason-model-foundry)."""

    def list_custom_models(self, domain: Optional[str] = None) -> List[ModelDefinition]:
        """Lists custom models available from the foundry, optionally filtered by domain.

        Args:
            domain: Optional domain filter (e.g., "medical").

        Returns:
            A list of ModelDefinition objects.
        """
        ...
