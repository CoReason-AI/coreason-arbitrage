# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from coreason_arbitrage.utils.logger import logger


def hello_world() -> str:
    """Returns a hello world message.

    This function serves as a basic health check and example of a public function.

    Returns:
        str: A string containing "Hello World!".
    """
    logger.info("Hello World!")
    return "Hello World!"
