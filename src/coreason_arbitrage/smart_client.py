# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

import os
from typing import Any, Dict, List, Optional, Set

from litellm import completion
from litellm.exceptions import APIConnectionError, RateLimitError, ServiceUnavailableError

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.gatekeeper import Gatekeeper
from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.router import Router
from coreason_arbitrage.utils.logger import logger

MAX_RETRIES = 3
RETRIABLE_ERRORS = (RateLimitError, ServiceUnavailableError, APIConnectionError)


class CompletionsWrapper:
    """Proxy for chat.completions.

    Handles the core logic of classification, routing, execution, and failover.
    """

    def __init__(self, engine: ArbitrageEngine, gatekeeper: Gatekeeper) -> None:
        self.engine = engine
        self.gatekeeper = gatekeeper

        if self.engine.budget_client is None:
            logger.warning("ArbitrageEngine not configured with BudgetClient. Router might fail.")

        # Inject LoadBalancer into Router for dynamic health checks
        # We ignore type for budget_client as it might be None but we handle it in Router or assume configured
        self.router = Router(
            self.engine.registry,
            self.engine.budget_client,  # type: ignore
            self.engine.load_balancer,
        )

    def create(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Orchestrates the Classify-Route-Execute loop.

        This method performs the following steps:
        1. Checks User Budget.
        2. Classifies the prompt (Gatekeeper).
        3. Routes to the optimal model (Router).
        4. Executes the request via `litellm`.
        5. Logs the transaction and deducts funds.

        It implements retry logic with provider exclusion and Fail-Open behavior.

        Args:
            messages: A list of message dictionaries (role, content).
            **kwargs: Additional arguments passed to `litellm.completion`.
                Note: The `model` argument is determined by the Router and should
                generally not be passed by the user, or it will be ignored/overwritten.

        Returns:
            The response object from the LLM provider.

        Raises:
            PermissionError: If budget check fails or funds are insufficient.
            RuntimeError: If routing fails or Fail-Open also fails.
        """
        user_id = kwargs.get("user", "default_user")

        # 0. Budget Check (Pre-flight)
        if self.engine.budget_client:
            try:
                if not self.engine.budget_client.check_allowance(user_id):
                    logger.warning(f"Budget exceeded for user {user_id}. Denying request.")
                    raise PermissionError("Budget exceeded.")
            except PermissionError:
                raise
            except Exception as e:
                # Fail Closed: If DB down, deny.
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
        routing_context = self.gatekeeper.classify(prompt)
        logger.info(f"Classified request: {routing_context}")

        # Retry Loop
        last_exception: Optional[Exception] = None
        failed_providers: Set[str] = set()

        for attempt in range(MAX_RETRIES):
            try:
                # 3. Routing (Inside loop to pick up new healthy models)
                model_def: ModelDefinition = self.router.route(
                    routing_context, user_id, excluded_providers=list(failed_providers)
                )
                logger.info(f"Selected model (Attempt {attempt + 1}): {model_def.id} ({model_def.provider})")

                # 4. Execution
                response = completion(model=model_def.id, messages=messages, **kwargs)

                # Record Success
                self.engine.load_balancer.record_success(model_def.provider)

                # 5. Cost Calculation & Accounting
                try:
                    usage = response.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens

                    # Calculate cost (simplified)
                    cost = (input_tokens / 1000 * model_def.cost_per_1k_input) + (
                        output_tokens / 1000 * model_def.cost_per_1k_output
                    )

                    # Audit Logging
                    if self.engine.audit_client:
                        try:
                            self.engine.audit_client.log_transaction(
                                user_id=user_id,
                                model_id=model_def.id,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cost=cost,
                            )
                        except Exception as e:
                            logger.error(f"Audit logging failed: {e}")

                    # Budget Deduction
                    if self.engine.budget_client:
                        try:
                            self.engine.budget_client.deduct_funds(user_id, cost)
                        except Exception as e:
                            # Log error but do not fail the request since response is already generated
                            logger.error(f"Failed to deduct funds for user {user_id}: {e}")
                except Exception as e:
                    logger.error(
                        f"Accounting/Cost Calculation failed: {e}. Skipping accounting but returning response."
                    )

                return response

            except RuntimeError as e:
                # Routing failed (e.g. no healthy models)
                logger.error(f"Routing failed on attempt {attempt + 1}: {e}")
                last_exception = e
                # If routing fails, it usually means no healthy models. Retrying immediately might not help
                # unless a model recovers in milliseconds. But we respect MAX_RETRIES.
                continue

            except Exception as e:
                # Execution failed
                logger.error(f"Execution failed on attempt {attempt + 1}: {e}")
                last_exception = e

                # Update LoadBalancer and Exclude Provider
                if "model_def" in locals():
                    # Only record failures for specific availability errors or if specifically requested
                    # And exclude provider from subsequent attempts in this request
                    if isinstance(e, RETRIABLE_ERRORS):
                        self.engine.load_balancer.record_failure(model_def.provider)
                        failed_providers.add(model_def.provider)
                        logger.warning(
                            f"Provider {model_def.provider} failed with critical error. Excluding from retry."
                        )
                    else:
                        # For other errors, we might still record generic failure but maybe not exclude immediately?
                        # User spec: "On a relevant error (5xx/429), add the current model's provider to that set"
                        # But for safety in failover logic, if a model fails execution,
                        # we generally want to try another.
                        # However, strictly following the instruction to "Restrict LoadBalancer to ... 5xx/429":
                        # We only record failure for those.
                        # Do we exclude for others?
                        # If it's a BadRequestError, switching provider might not help (bad prompt).
                        # So we stick to excluding only on relevant errors.
                        pass

                # Continue to next attempt
                continue

        # If exhausted retries, Fail Open
        logger.critical(f"Max retries exhausted. Fail-Open triggered. Last error: {last_exception}")

        fallback_model_id = os.environ.get("ARBITRAGE_FALLBACK_MODEL", "azure/gpt-4o")
        logger.warning(f"Attempting Fail-Open with model: {fallback_model_id}")

        # Construct safe default model definition for audit/tracking
        # We assume standard GPT-4o pricing or $0 if unknown
        fallback_model = ModelDefinition(
            id=fallback_model_id,
            provider="failover",
            tier=ModelTier.TIER_3_REASONING,
            cost_per_1k_input=0.005,  # Estimated
            cost_per_1k_output=0.015,
            is_healthy=True,
        )

        try:
            response = completion(model=fallback_model.id, messages=messages, **kwargs)

            # Record success (maybe not for load balancer since provider is fake/failover)
            # But we should log audit and deduct funds

            try:
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                cost = (input_tokens / 1000 * fallback_model.cost_per_1k_input) + (
                    output_tokens / 1000 * fallback_model.cost_per_1k_output
                )

                if self.engine.audit_client:
                    try:
                        self.engine.audit_client.log_transaction(
                            user_id=user_id,
                            model_id=fallback_model.id,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cost=cost,
                        )
                    except Exception as e:
                        logger.error(f"Audit logging failed during fail-open: {e}")

                if self.engine.budget_client:
                    try:
                        self.engine.budget_client.deduct_funds(user_id, cost)
                    except Exception as e:
                        logger.error(f"Failed to deduct funds during fail-open for user {user_id}: {e}")
            except Exception as e:
                logger.error(
                    f"Accounting/Cost Calculation failed during fail-open: {e}. "
                    "Skipping accounting but returning response."
                )

            return response

        except Exception as e:
            logger.critical(f"Fail-Open failed: {e}")
            if last_exception:
                raise last_exception from e
            raise e from None


class ChatWrapper:
    """Proxy for chat namespace."""

    def __init__(self, engine: ArbitrageEngine) -> None:
        self.completions = CompletionsWrapper(engine, Gatekeeper())


class SmartClient:
    """SmartClient proxy class mimicking OpenAI client interface.

    This client provides an interface compatible with standard OpenAI libraries
    while abstracting away the underlying provider selection and failover logic.
    """

    def __init__(self, engine: ArbitrageEngine) -> None:
        self.chat = ChatWrapper(engine)
