from unittest.mock import MagicMock

import pytest
from coreason_identity.models import UserContext


@pytest.fixture
def user_context() -> UserContext:
    uc = MagicMock(spec=UserContext)
    uc.user_id = "default_user"
    uc.groups = ["users"]
    return uc
