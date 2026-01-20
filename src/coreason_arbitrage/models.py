# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    """Enumeration of model capability tiers."""

    TIER_1_FAST = "fast"
    TIER_2_SMART = "smart"
    TIER_3_REASONING = "reasoning"


class ModelDefinition(BaseModel):
    """Defines the properties and state of an LLM model."""

    id: str = Field(..., min_length=1)  # e.g. "azure/gpt-4o"
    provider: str = Field(..., min_length=1)  # e.g. "azure"
    tier: ModelTier
    cost_per_1k_input: float
    cost_per_1k_output: float
    is_healthy: bool = True
    domain: Optional[str] = None


class RoutingContext(BaseModel):
    """Context derived from the user's prompt to guide routing decisions."""

    complexity: float = Field(..., ge=0.0, le=1.0)
    domain: Optional[str] = None


class RoutingPolicy(BaseModel):
    """Configuration for a routing policy (e.g., loaded from YAML)."""

    name: str
    condition: str
    models: List[str]
    fallback: List[str]
