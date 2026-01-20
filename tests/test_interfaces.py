# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from typing import List, Optional

from coreason_arbitrage.interfaces import AuditClient, BudgetClient, ModelFoundryClient
from coreason_arbitrage.models import ModelDefinition, ModelTier


# --- Mocks for Positive Tests ---
class MockBudgetClient:
    def check_allowance(self, user_id: str) -> bool:
        return True

    def get_remaining_budget_percentage(self, user_id: str) -> float:
        return 0.5

    def deduct_funds(self, user_id: str, amount: float) -> None:
        pass


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


# --- Classes for Negative/Edge Case Tests ---
class IncompleteBudgetClient:
    """Missing get_remaining_budget_percentage"""

    def check_allowance(self, user_id: str) -> bool:
        return True


class EmptyClass:
    pass


class BaseBudgetImpl:
    """Implements one method"""

    def check_allowance(self, user_id: str) -> bool:
        return True


class CompleteBudgetImpl(BaseBudgetImpl):
    """Implements the missing method, completing the protocol via inheritance"""

    def get_remaining_budget_percentage(self, user_id: str) -> float:
        return 0.8

    def deduct_funds(self, user_id: str, amount: float) -> None:
        pass


# --- Tests ---


def test_budget_client_protocol_positive() -> None:
    # Verify that MockBudgetClient adheres to the BudgetClient protocol
    client = MockBudgetClient()
    assert isinstance(client, BudgetClient)
    assert client.check_allowance("user1") is True
    assert client.get_remaining_budget_percentage("user1") == 0.5


def test_budget_client_protocol_negative() -> None:
    # Verify incomplete implementations do not satisfy the protocol
    incomplete = IncompleteBudgetClient()
    assert not isinstance(incomplete, BudgetClient)

    empty = EmptyClass()
    assert not isinstance(empty, BudgetClient)


def test_budget_client_complex_inheritance() -> None:
    # Verify that a subclass completing the protocol is recognized
    # Base class only implements part of it
    base = BaseBudgetImpl()
    assert not isinstance(base, BudgetClient)

    # Subclass implements the rest
    complete = CompleteBudgetImpl()
    assert isinstance(complete, BudgetClient)
    assert complete.check_allowance("u1") is True
    assert complete.get_remaining_budget_percentage("u1") == 0.8


def test_audit_client_protocol() -> None:
    client = MockAuditClient()
    assert isinstance(client, AuditClient)
    # Check that it works (no exceptions)
    client.log_transaction("user1", "model1", 10, 20, 0.05)


def test_audit_client_negative() -> None:
    empty = EmptyClass()
    assert not isinstance(empty, AuditClient)


def test_model_foundry_client_protocol() -> None:
    client = MockModelFoundryClient()
    assert isinstance(client, ModelFoundryClient)
    models = client.list_custom_models()
    assert len(models) == 1
    assert models[0].id == "custom/model-1"


def test_model_foundry_client_negative() -> None:
    empty = EmptyClass()
    assert not isinstance(empty, ModelFoundryClient)
