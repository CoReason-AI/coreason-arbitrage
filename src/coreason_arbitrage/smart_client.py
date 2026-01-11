# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from typing import Any, Dict, List, Optional

from litellm import completion

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.gatekeeper import Gatekeeper
from coreason_arbitrage.models import ModelDefinition
from coreason_arbitrage.router import Router
from coreason_arbitrage.utils.logger import logger

MAX_RETRIES = 3


class CompletionsWrapper:
    """
    Proxy for chat.completions.
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
        """
        Orchestrates the Classify-Route-Execute loop.
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
        for attempt in range(MAX_RETRIES):
            try:
                # 3. Routing (Inside loop to pick up new healthy models)
                model_def: ModelDefinition = self.router.route(routing_context, user_id)
                logger.info(f"Selected model (Attempt {attempt + 1}): {model_def.id} ({model_def.provider})")

                # 4. Execution
                response = completion(model=model_def.id, messages=messages, **kwargs)

                # Record Success
                self.engine.load_balancer.record_success(model_def.provider)

                # 5. Audit Logging
                if self.engine.audit_client:
                    try:
                        usage = response.usage
                        input_tokens = usage.prompt_tokens
                        output_tokens = usage.completion_tokens

                        # Calculate cost (simplified)
                        cost = (input_tokens / 1000 * model_def.cost_per_1k_input) + (
                            output_tokens / 1000 * model_def.cost_per_1k_output
                        )

                        self.engine.audit_client.log_transaction(
                            user_id=user_id,
                            model_id=model_def.id,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cost=cost,
                        )
                    except Exception as e:
                        logger.error(f"Audit logging failed: {e}")

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

                # Update LoadBalancer
                if "model_def" in locals():
                    self.engine.load_balancer.record_failure(model_def.provider)

                # Continue to next attempt
                continue

        # If exhausted retries
        logger.error("Max retries exhausted.")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Request failed after max retries.")


class ChatWrapper:
    """
    Proxy for chat namespace.
    """

    def __init__(self, engine: ArbitrageEngine) -> None:
        self.completions = CompletionsWrapper(engine, Gatekeeper())


class SmartClient:
    """
    SmartClient proxy class mimicking OpenAI client interface.
    """

    def __init__(self, engine: ArbitrageEngine) -> None:
        self.chat = ChatWrapper(engine)
