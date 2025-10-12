# https://github.com/stanfordnlp/dspy/blob/main/dspy/dsp/utils/settings.py

import asyncio
import contextvars
import copy
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

DEFAULT_CONFIG = dict(
    local_lm=None,
)

# Global base configuration and owner tracking
main_thread_config = copy.deepcopy(DEFAULT_CONFIG)
config_owner_thread_id = None
config_owner_async_task = None

# Global lock for settings configuration
global_lock = threading.Lock()

thread_local_overrides: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "context_overrides", default=None
)


class Settings:
    """A singleton class for DSPy configuration settings.
    Thread-safe global configuration.
    - 'configure' can be called by only one 'owner' thread (the first thread that calls it).
    - Other threads see the configured global values from 'main_thread_config'.
    - 'context' sets thread-local overrides. These overrides propagate to threads spawned
      inside that context block, when (and only when!) using a ParallelExecutor that copies overrides.

      1. Only one unique thread (which can be any thread!) can call dspy.configure.
      2. It affects a global state, visible to all. As a result, user threads work, but they shouldn't be
         mixed with concurrent changes to dspy.configure from the "main" thread.
         (TODO: In the future, add warnings: if there are near-in-time user-thread reads followed by .configure calls.)
      3. Any thread can use dspy.context. It propagates to child threads
         created with DSPy primitives: Parallel, asyncify, etc.
    """

    _instance = None

    def __new__(cls) -> "Settings":  # noqa: D102
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def lock(self) -> threading.Lock:  # noqa: D102
        return global_lock

    def __getattr__(self, name: str) -> Any:  # noqa: D105, ANN401
        overrides = thread_local_overrides.get() or {}
        if name in overrides:
            return overrides[name]
        elif name in main_thread_config:
            return main_thread_config[name]
        else:
            raise AttributeError(f"'Settings' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: D105, ANN401
        if name in {"_instance"}:
            super().__setattr__(name, value)
        else:
            self.configure(**{name: value})

    def __getitem__(self, key: str) -> Any:  # noqa: D105, ANN401
        return self.__getattr__(key)

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: D105, ANN401
        self.__setattr__(key, value)

    def __contains__(self, key: str) -> bool:  # noqa: D105
        overrides = thread_local_overrides.get() or {}
        return key in overrides or key in main_thread_config

    def get(self, key: str, default: Any = None) -> Any:  # noqa: D102, ANN401
        try:
            return self[key]
        except AttributeError:
            return default

    def copy(self) -> dict[str, Any]:  # noqa: D102, PLR6301
        overrides = thread_local_overrides.get() or {}
        return dict({**main_thread_config, **overrides})

    @property
    def config(self) -> dict[str, Any]:  # noqa: D102
        return self.copy()

    def _ensure_configure_allowed(self) -> None:  # noqa: PLR6301
        global config_owner_thread_id, config_owner_async_task  # noqa: PLW0603
        current_thread_id = threading.get_ident()

        if config_owner_thread_id is None:
            # First `configure` call assigns the owner thread id.
            config_owner_thread_id = current_thread_id

        if config_owner_thread_id != current_thread_id:
            # Disallow a second `configure` calls from other threads.
            raise RuntimeError("dspy.settings can only be changed by the thread that initially configured it.")

        # Async task doesn't allow a second `configure` call, must use dspy.context(...) instead.
        is_async_task = False
        try:
            if asyncio.current_task() is not None:
                is_async_task = True
        except RuntimeError:
            # This exception (e.g., "no current task") means we are not in an async loop/task,
            # or asyncio module itself is not fully functional in this specific sub-thread context.
            is_async_task = False

        if not is_async_task:
            return

        if config_owner_async_task is None:
            # First `configure` call assigns the owner async task.
            config_owner_async_task = asyncio.current_task()
            return

        # We are in an async task. Now check for IPython and allow calling `configure` from IPython.
        in_ipython = False
        try:
            from IPython import get_ipython  # noqa: PLC0415

            # get_ipython is a global injected by IPython environments.
            # We check its existence and type to be more robust.
            in_ipython = get_ipython() is not None
        except Exception:  # noqa: BLE001
            # If `IPython` is not installed or `get_ipython` failed, we are not in an IPython environment.
            in_ipython = False

        if not in_ipython and config_owner_async_task != asyncio.current_task():
            raise RuntimeError(
                "dspy.settings.configure(...) can only be called from the same async task that called it first. Please "
                "use `dspy.context(...)` in other async tasks instead."
            )

    def configure(self, **kwargs: Any) -> None:  # noqa: D102, ANN401
        # If no exception is raised, the `configure` call is allowed.
        self._ensure_configure_allowed()

        # Update global config
        for k, v in kwargs.items():
            main_thread_config[k] = v

    @contextmanager
    def context(self, **kwargs: Any) -> Generator[None, None, None]:  # noqa: ANN401, PLR6301
        """Context manager for temporary configuration changes at the thread level.
        Does not affect global configuration. Changes only apply to the current thread.
        If threads are spawned inside this block using ParallelExecutor, they will inherit these overrides.
        """
        original_overrides = (thread_local_overrides.get() or {}).copy()
        new_overrides = dict({**main_thread_config, **original_overrides, **kwargs})
        token = thread_local_overrides.set(new_overrides)

        try:
            yield
        finally:
            thread_local_overrides.reset(token)

    def __repr__(self) -> str:  # noqa: D105
        overrides = thread_local_overrides.get() or {}
        combined_config = {**main_thread_config, **overrides}
        return repr(combined_config)


settings = Settings()
