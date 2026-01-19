# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

"""Coreason Arbitrage Package.

Intelligent routing layer for LLMs, optimizing for cost, performance, and reliability.
"""

__version__ = "0.1.0"
__author__ = "Gowtham A Rao"
__email__ = "gowtham.rao@coreason.ai"

from .engine import ArbitrageEngine
from .main import hello_world
from .smart_client import SmartClient, SmartClientAsync

__all__ = ["ArbitrageEngine", "SmartClient", "SmartClientAsync", "hello_world"]
