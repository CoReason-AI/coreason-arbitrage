import threading
from typing import Dict, List, Optional

from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.utils.logger import logger


class ModelRegistry:
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
        """
        Registers a model in the registry.
        If a model with the same ID exists, it is updated.
        """
        with self._lock:
            self._models[model.id] = model
            logger.debug(f"Registered model: {model.id} (Tier: {model.tier})")

    def get_model(self, model_id: str) -> Optional[ModelDefinition]:
        """
        Retrieves a model by its ID.
        """
        return self._models.get(model_id)

    def list_models(self, tier: Optional[ModelTier] = None) -> List[ModelDefinition]:
        """
        Lists all models, optionally filtered by tier.
        """
        with self._lock:
            all_models = list(self._models.values())

        if tier:
            return [m for m in all_models if m.tier == tier]
        return all_models

    def clear(self) -> None:
        """
        Clears the registry (useful for testing).
        """
        with self._lock:
            self._models.clear()
            logger.debug("ModelRegistry cleared")
