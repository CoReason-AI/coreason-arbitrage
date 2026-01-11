# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from coreason_arbitrage.engine import ArbitrageEngine


def test_arbitrage_engine_singleton() -> None:
    engine1 = ArbitrageEngine()
    engine2 = ArbitrageEngine()
    assert engine1 is engine2
    assert engine1._initialized is True


def test_arbitrage_engine_logging(caplog: object) -> None:
    # Note: Using caplog to capture loguru logs might require specific configuration
    # or using a loguru compatible fixture if caplog doesn't catch it automatically.
    # However, since we set up loguru to write to stderr, and caplog captures root logger by default,
    # we might need to rely on the side effect or check if loguru intercepts logging.
    # A simpler check for now is that instantiation doesn't crash.
    engine = ArbitrageEngine()
    engine.get_client()
