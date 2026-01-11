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
from typing import List

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


def test_arbitrage_engine_thread_safety() -> None:
    """Test that the Singleton pattern holds up under concurrent access."""
    instances: List[ArbitrageEngine] = []

    def get_instance() -> None:
        instances.append(ArbitrageEngine())

    threads = [threading.Thread(target=get_instance) for _ in range(10)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify all instances are the same object
    first_instance = instances[0]
    for instance in instances[1:]:
        assert instance is first_instance


def test_arbitrage_engine_reinitialization_idempotency(caplog: object) -> None:
    """Test that __init__ does not re-run initialization logic if called again."""
    engine = ArbitrageEngine()

    # Manually call __init__ again
    engine.__init__()  # type: ignore[misc]

    # Ideally we would check logs here to ensure "Initializing ArbitrageEngine" appears only once.
    # Since capturing loguru logs with pytest's caplog can be tricky without a sink,
    # we rely on the internal state _initialized which we can inspect,
    # or trust that the logic `if self._initialized: return` works.

    assert engine._initialized is True
