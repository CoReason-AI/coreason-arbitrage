# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from coreason_identity.models import UserContext
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.smart_client import SmartClientAsync
from coreason_arbitrage.utils.logger import logger

# --- Mock Clients ---


class MockBudgetClient:
    def check_allowance(self, user_id: str) -> bool:
        return True

    def get_remaining_budget_percentage(self, user_context: UserContext) -> float:
        return 1.0

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
        logger.info(f"Audit Log: User={user_id}, Model={model_id}, Cost={cost}")


class MockFoundryClient:
    def list_custom_models(self, domain: Optional[str] = None) -> List[ModelDefinition]:
        return [
            ModelDefinition(
                id="azure/gpt-4o",
                provider="azure",
                tier=ModelTier.TIER_1_FAST,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.03,
                is_healthy=True,
            )
        ]


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting up coreason-arbitrage server...")
    engine = ArbitrageEngine()

    # Configure with mocks for now
    engine.configure(
        budget_client=MockBudgetClient(),
        audit_client=MockAuditClient(),
        foundry_client=MockFoundryClient(),
    )

    app.state.engine = engine
    yield
    logger.info("Shutting down coreason-arbitrage server...")


# --- App ---

app = FastAPI(title="Coreason Arbitrage", lifespan=lifespan)


# --- Models ---


class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = None  # Optional, will be overridden by router often
    user: str = Field(..., description="The user ID")
    temperature: Optional[float] = 0.7

    model_config = ConfigDict(extra="allow")


# --- Endpoints ---


@app.get("/health")  # type: ignore
async def health_check() -> Dict[str, str]:
    return {"status": "ready", "routing_engine": "active"}


@app.post("/v1/chat/completions")  # type: ignore
async def chat_completions(request: ChatCompletionRequest) -> Any:
    engine: ArbitrageEngine = app.state.engine

    # Convert request to dict, excluding user which is passed separately if needed
    request_dict = request.model_dump(exclude_none=True)
    messages = request_dict.pop("messages")
    user_id = request_dict.pop("user")

    # The rest are passed as kwargs to create, which passes them to litellm.acompletion
    # SmartClientAsync.create expects 'user' in kwargs if UserContext is not provided.

    try:
        async with SmartClientAsync(engine) as client:
            response = await client.chat.completions.create(messages=messages, user=user_id, **request_dict)
            return response

    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e)) from e
    except RuntimeError as e:
        # Routing failed
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
