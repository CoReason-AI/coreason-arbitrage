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
from typing import Dict, List, Optional

from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.utils.logger import logger


class ModelRegistry:
    """Singleton registry for storing and retrieving model definitions.

    Thread-safe storage for model configurations, supporting lookups by ID,
    tier, and domain.
    """

    _instance: Optional["ModelRegistry"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelRegistry, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        with self._lock:
            # Check again (double-checked locking)
            if self._initialized:  # pragma: no cover
                return
            self._models: Dict[str, ModelDefinition] = {}
            self._initialized = True
            logger.info("ModelRegistry initialized")

    def register_model(self, model: ModelDefinition) -> None:
        """Registers a model in the registry.

        If a model with the same ID exists, it is overwritten.

        Args:
            model: The ModelDefinition object to register.
        """
        with self._lock:
            self._models[model.id] = model
            logger.debug(f"Registered model: {model.id} (Tier: {model.tier})")

    def get_model(self, model_id: str) -> Optional[ModelDefinition]:
        """Retrieves a model by its ID.

        Args:
            model_id: The unique identifier of the model.

        Returns:
            The ModelDefinition if found, otherwise None.
        """
        return self._models.get(model_id)

    def list_models(self, tier: Optional[ModelTier] = None, domain: Optional[str] = None) -> List[ModelDefinition]:
        """Lists all models, optionally filtered by tier and/or domain.

        Domain matching is case-insensitive.

        Args:
            tier: Optional ModelTier to filter by.
            domain: Optional domain string to filter by.

        Returns:
            A list of matching ModelDefinition objects.
        """
        with self._lock:
            all_models = list(self._models.values())

        filtered = all_models
        if tier:
            filtered = [m for m in filtered if m.tier == tier]

        if domain:
            domain_lower = domain.lower()
            filtered = [m for m in filtered if m.domain and m.domain.lower() == domain_lower]

        return filtered

    def clear(self) -> None:
        """Clears the registry.

        This is primarily used for testing purposes to reset state.
        """
        with self._lock:
            self._models.clear()
            logger.debug("ModelRegistry cleared")
