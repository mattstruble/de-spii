import os
from collections.abc import Callable, Generator
from typing import Any
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _reset_environment() -> Generator[Any, Any, Any]:
    """Automatically reset the os environment after every test."""
    init_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(init_env)


@pytest.fixture
def ctx_factory() -> Callable[[str], mock.Mock]:
    def _factory(text: str) -> mock.Mock:
        ctx = mock.Mock()
        ctx.text = text
        ctx.redact = mock.Mock()
        return ctx

    return _factory
