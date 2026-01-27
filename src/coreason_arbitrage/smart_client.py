# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

import functools
import os
from typing import Any, Dict, List, Optional, Set

import anyio
import httpx
from anyio.from_thread import start_blocking_portal
from coreason_identity.models import UserContext
from litellm import acompletion
from litellm.exceptions import APIConnectionError, RateLimitError, ServiceUnavailableError

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.gatekeeper import Gatekeeper
from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.router import Router
from coreason_arbitrage.utils.logger import logger

MAX_RETRIES = 3
RETRIABLE_ERRORS = (RateLimitError, ServiceUnavailableError, APIConnectionError)


class CompletionsWrapperAsync:
    """Async Proxy for chat.completions.

    Handles the core logic of classification, routing, execution, and failover.
    """

    def __init__(
        self,
        engine: ArbitrageEngine,
        gatekeeper: Gatekeeper,
        client: Optional[httpx.AsyncClient],
    ) -> None:
        self.engine = engine
        self.gatekeeper = gatekeeper
        self._client = client

        if self.engine.budget_client is None:
            logger.warning("ArbitrageEngine not configured with BudgetClient. Router might fail.")

        self.router = Router(
            self.engine.registry,
            self.engine.budget_client,  # type: ignore
            self.engine.load_balancer,
        )

    async def create(
        self,
        messages: List[Dict[str, str]],
        user_context: Optional[UserContext] = None,
        **kwargs: Any,
    ) -> Any:
        """Orchestrates the Classify-Route-Execute loop asynchronously.

        Args:
            messages: A list of message dictionaries (role, content).
            user_context: The authenticated user context.
            **kwargs: Additional arguments passed to `litellm.acompletion`.

        Returns:
            The response object from the LLM provider.

        Raises:
            PermissionError: If budget check fails or funds are insufficient.
            RuntimeError: If routing fails or Fail-Open also fails.
        """
        # Determine User ID for accounting/logging
        if user_context:
            user_id = user_context.user_id
        else:
            # Legacy fallback: check kwargs for 'user'
            user_id = kwargs.get("user", "default_user")
            if "user" in kwargs:
                # If 'user' was passed but no context, we rely on Router's fail safe
                pass

        # 0. Budget Check (Pre-flight)
        if self.engine.budget_client:
            try:
                allowed = await anyio.to_thread.run_sync(self.engine.budget_client.check_allowance, user_id)
                if not allowed:
                    logger.warning(f"Budget exceeded for user {user_id}. Denying request.")
                    raise PermissionError("Budget exceeded.")
            except PermissionError:
                raise
            except Exception as e:
                logger.error(f"Budget check failed: {e}. Failing Closed.")
                raise PermissionError("Budget check failed.") from e

        # 1. Extract prompt for classification
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        if not prompt:
            logger.warning("No user message found in messages list. Using empty string for classification.")

        # 2. Gatekeeper Classification
        # Regex is fast, so running on main thread is fine.
        routing_context = self.gatekeeper.classify(prompt)
        logger.info(f"Classified request: {routing_context}")

        # Retry Loop
        last_exception: Optional[Exception] = None
        failed_providers: Set[str] = set()

        for attempt in range(MAX_RETRIES):
            try:
                # 3. Routing (Inside loop to pick up new healthy models)
                model_def: ModelDefinition = self.router.route(
                    routing_context,
                    user_context=user_context,
                    excluded_providers=list(failed_providers),
                )
                logger.info(f"Selected model (Attempt {attempt + 1}): {model_def.id} ({model_def.provider})")

                # 4. Execution
                response = await acompletion(model=model_def.id, messages=messages, **kwargs)

                # Record Success
                self.engine.load_balancer.record_success(model_def.provider)

                # 5. Cost Calculation & Accounting
                await self._handle_accounting(user_id, model_def, response)

                return response

            except RuntimeError as e:
                # Routing failed (e.g. no healthy models)
                logger.error(f"Routing failed on attempt {attempt + 1}: {e}")
                last_exception = e
                continue

            except Exception as e:
                # Execution failed
                logger.error(f"Execution failed on attempt {attempt + 1}: {e}")
                last_exception = e

                if "model_def" in locals() and isinstance(e, RETRIABLE_ERRORS):
                    self.engine.load_balancer.record_failure(model_def.provider)
                    failed_providers.add(model_def.provider)
                    logger.warning(f"Provider {model_def.provider} failed with critical error. Excluding from retry.")

                continue

        # If exhausted retries, Fail Open
        logger.critical(f"Max retries exhausted. Fail-Open triggered. Last error: {last_exception}")
        return await self._fail_open(messages, user_id, last_exception, **kwargs)

    async def _handle_accounting(self, user_id: str, model_def: ModelDefinition, response: Any) -> None:
        try:
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

            cost = (input_tokens / 1000 * model_def.cost_per_1k_input) + (
                output_tokens / 1000 * model_def.cost_per_1k_output
            )

            # Audit Logging
            if self.engine.audit_client:
                try:
                    func = functools.partial(
                        self.engine.audit_client.log_transaction,
                        user_id=user_id,
                        model_id=model_def.id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                    )
                    await anyio.to_thread.run_sync(func)
                except Exception as e:
                    logger.error(f"Audit logging failed: {e}")

            # Budget Deduction
            if self.engine.budget_client:
                try:
                    func = functools.partial(self.engine.budget_client.deduct_funds, user_id=user_id, amount=cost)
                    await anyio.to_thread.run_sync(func)
                except Exception as e:
                    logger.error(f"Failed to deduct funds for user {user_id}: {e}")
        except Exception as e:
            logger.error(f"Accounting/Cost Calculation failed: {e}. Skipping accounting but returning response.")

    async def _fail_open(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        last_exception: Optional[Exception],
        **kwargs: Any,
    ) -> Any:
        fallback_model_id = os.environ.get("ARBITRAGE_FALLBACK_MODEL", "azure/gpt-4o")
        logger.warning(f"Attempting Fail-Open with model: {fallback_model_id}")

        fallback_model = ModelDefinition(
            id=fallback_model_id,
            provider="failover",
            tier=ModelTier.TIER_3_REASONING,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            is_healthy=True,
        )

        try:
            response = await acompletion(model=fallback_model.id, messages=messages, **kwargs)
            await self._handle_accounting(user_id, fallback_model, response)
            return response

        except Exception as e:
            logger.critical(f"Fail-Open failed: {e}")
            if last_exception:
                raise last_exception from e
            raise e from None


class ChatWrapperAsync:
    """Async Proxy for chat namespace."""

    def __init__(self, engine: ArbitrageEngine, client: Optional[httpx.AsyncClient]) -> None:
        self.completions = CompletionsWrapperAsync(engine, Gatekeeper(), client)


class SmartClientAsync:
    """Async SmartClient proxy class.

    Handles connection lifecycle and async execution.
    """

    def __init__(self, engine: ArbitrageEngine, client: Optional[httpx.AsyncClient] = None) -> None:
        self.engine = engine
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()
        self.chat = ChatWrapperAsync(engine, self._client)

    async def __aenter__(self) -> "SmartClientAsync":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._internal_client:
            await self._client.aclose()

    async def close(self) -> None:
        """Closes the underlying HTTP client."""
        if self._internal_client:
            await self._client.aclose()


class CompletionsWrapper:
    """Sync Facade for chat.completions."""

    def __init__(self, client: "SmartClient") -> None:
        self._client = client

    @property
    def router(self) -> Router:
        """Exposes the underlying router."""
        # If we have an async instance (Context Manager), use its router
        if self._client._async_instance:
            return self._client._async_instance.chat.completions.router
        # Fallback for legacy/non-context usage: Use client's fallback router
        return self._client._router_fallback

    @router.setter
    def router(self, value: Router) -> None:
        if self._client._async_instance:
            self._client._async_instance.chat.completions.router = value
        else:
            self._client._router_fallback = value

    def create(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Synchronous wrapper for create."""
        return self._client._create_completion_sync(messages, **kwargs)


class ChatWrapper:
    """Sync Facade for chat namespace."""

    def __init__(self, client: "SmartClient") -> None:
        self.completions = CompletionsWrapper(client)


class SmartClient:
    """Sync Facade for SmartClientAsync.

    Provides a blocking interface compatible with legacy code.
    Supports usage as a Context Manager for connection pooling.
    """

    def __init__(self, engine: ArbitrageEngine) -> None:
        self.engine = engine
        self.chat = ChatWrapper(self)

        # State for Context Manager usage
        self._portal_ctx: Any = None
        self._portal: Any = None
        self._async_instance: Optional[SmartClientAsync] = None

        # Router for fallback usage (legacy tests)
        if self.engine.budget_client is None:
            logger.warning("ArbitrageEngine not configured with BudgetClient. Router might fail.")
        self._router_fallback = Router(
            self.engine.registry,
            self.engine.budget_client,  # type: ignore
            self.engine.load_balancer,
        )

    def _create_completion_sync(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        if self._portal and self._async_instance:
            # Use persistent portal and client (Pooling)
            return self._portal.call(self._async_instance.chat.completions.create, messages, **kwargs)
        else:
            # Fallback: One-off Loop (Safe, no pooling)
            async def _one_off() -> Any:
                async with SmartClientAsync(self.engine) as svc:
                    # Sync router fallback state if modified in legacy tests?
                    # Tests verify setting client.chat.completions.router
                    # If user set _router_fallback, we should probably use it?
                    # SmartClientAsync creates its own Router.
                    # We can inject _router_fallback into svc if needed.
                    # svc.chat.completions.router = self._router_fallback
                    # This ensures mocks on router work.
                    svc.chat.completions.router = self._router_fallback
                    return await svc.chat.completions.create(messages, **kwargs)

            return anyio.run(_one_off)

    def __enter__(self) -> "SmartClient":
        self._portal_ctx = start_blocking_portal()
        self._portal = self._portal_ctx.__enter__()

        # Instantiate Async Client inside the portal logic?
        # Actually we can instantiate it here, but httpx client MUST NOT be used until inside loop.
        # SmartClientAsync creates httpx.AsyncClient(). It is not bound until request.
        self._async_instance = SmartClientAsync(self.engine)

        # We should probably run __aenter__ of SmartClientAsync?
        # Yes, to be proper.
        self._portal.call(self._async_instance.__aenter__)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if self._async_instance:
                self._portal.call(self._async_instance.__aexit__, exc_type, exc_val, exc_tb)
        finally:
            if self._portal_ctx:
                self._portal_ctx.__exit__(exc_type, exc_val, exc_tb)
            self._portal = None
            self._async_instance = None

    def close(self) -> None:
        """Closes the underlying resources."""
        if self._portal and self._async_instance:
            self._portal.call(self._async_instance.close)
        elif self._portal is None:
            # If not in context, nothing to close (one-offs clean themselves up)
            pass
