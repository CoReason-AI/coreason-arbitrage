from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    TIER_1_FAST = "fast"
    TIER_2_SMART = "smart"
    TIER_3_REASONING = "reasoning"


class ModelDefinition(BaseModel):
    id: str  # e.g. "azure/gpt-4o"
    provider: str  # e.g. "azure"
    tier: ModelTier
    cost_per_1k_input: float
    cost_per_1k_output: float
    is_healthy: bool = True


class RoutingContext(BaseModel):
    complexity: float = Field(..., ge=0.0, le=1.0)
    domain: Optional[str] = None


class RoutingPolicy(BaseModel):
    name: str
    condition: str
    models: List[str]
    fallback: List[str]
