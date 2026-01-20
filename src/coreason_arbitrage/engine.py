# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

import threading
from typing import Optional

from coreason_arbitrage.interfaces import AuditClient, BudgetClient, ModelFoundryClient
from coreason_arbitrage.load_balancer import LoadBalancer
from coreason_arbitrage.registry import ModelRegistry
from coreason_arbitrage.utils.logger import logger


class ArbitrageEngine:
    """Singleton engine that orchestrates model routing, load balancing, and client management.

    This class serves as the central point of the coreason-arbitrage system, maintaining
    global state such as the LoadBalancer and ModelRegistry. It ensures that provider health
    and model configurations are shared across all client instances.
    """

    _instance: Optional["ArbitrageEngine"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> "ArbitrageEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ArbitrageEngine, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:  # Double check
                return  # pragma: no cover
            logger.info("Initializing ArbitrageEngine")
            self.load_balancer = LoadBalancer()
            self.registry = ModelRegistry()

            # Dependencies to be injected via configure
            self.budget_client: Optional[BudgetClient] = None
            self.audit_client: Optional[AuditClient] = None
            self.foundry_client: Optional[ModelFoundryClient] = None

            self._initialized = True

    def configure(
        self,
        budget_client: BudgetClient,
        audit_client: AuditClient,
        foundry_client: ModelFoundryClient,
    ) -> None:
        """Injects external dependencies into the engine and performs initial synchronization.

        Args:
            budget_client: Client implementation for budget checks and deductions.
            audit_client: Client implementation for transaction logging.
            foundry_client: Client implementation for fetching custom models.
        """
        with self._lock:
            self.budget_client = budget_client
            self.audit_client = audit_client
            self.foundry_client = foundry_client
            logger.info("ArbitrageEngine configured with external clients")

            # Initial sync with Model Foundry
            try:
                logger.info("Pulling custom models from ModelFoundry...")
                custom_models = self.foundry_client.list_custom_models()
                for model in custom_models:
                    self.registry.register_model(model)
                logger.info(f"Successfully registered {len(custom_models)} custom models from Foundry")
            except Exception as e:
                # Fail Open: Do not crash if Foundry is down, just log error
                logger.error(f"Failed to pull models from ModelFoundry: {e}")

    def get_client(self, capability: str = "reasoning") -> "SmartClient":  # type: ignore[name-defined] # noqa: F821
        """Returns a SmartClient instance configured for the engine.

        Args:
            capability: The desired capability (e.g., 'reasoning'). Currently unused
                but reserved for future specific client configuration.

        Returns:
            A SmartClient instance ready for making requests.
        """
        from coreason_arbitrage.smart_client import SmartClient

        logger.info(f"Getting client for capability: {capability}")
        return SmartClient(self)
