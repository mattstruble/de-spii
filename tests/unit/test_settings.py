import sys
import threading
from unittest.mock import Mock, patch

import pytest

from despii.settings import Settings

# Get the actual module object
settings_module = sys.modules["despii.settings"]


@pytest.fixture(autouse=True)
def _reset_settings():
    """Reset global settings state before and after each test."""
    # Store original values
    original_config = settings_module.main_thread_config.copy()
    original_owner_thread = settings_module.config_owner_thread_id
    original_owner_async = settings_module.config_owner_async_task
    original_overrides = settings_module.thread_local_overrides.get()

    # Reset to defaults
    settings_module.main_thread_config.clear()
    settings_module.main_thread_config.update(settings_module.DEFAULT_CONFIG.copy())
    settings_module.config_owner_thread_id = None
    settings_module.config_owner_async_task = None
    settings_module.thread_local_overrides.set(None)

    yield

    # Restore original values
    settings_module.main_thread_config.clear()
    settings_module.main_thread_config.update(original_config)
    settings_module.config_owner_thread_id = original_owner_thread
    settings_module.config_owner_async_task = original_owner_async
    if original_overrides is not None:
        settings_module.thread_local_overrides.set(original_overrides)
    else:
        settings_module.thread_local_overrides.set(None)


class TestSettingsSingleton:
    """Test Settings singleton behavior."""

    def test_singleton_returns_same_instance(self):
        """Test that Settings() always returns the same instance."""
        s1 = Settings()
        s2 = Settings()
        assert s1 is s2

    def test_singleton_with_imported_settings(self):
        """Test that imported settings is the singleton instance."""
        s = Settings()
        assert s is settings_module.settings


class TestSettingsLock:
    """Test Settings lock property."""

    def test_lock_returns_global_lock(self):
        """Test that lock property returns the global lock."""
        s = Settings()
        assert s.lock is settings_module.global_lock

    def test_lock_is_threading_lock(self):
        """Test that lock is a threading.Lock instance."""
        s = Settings()
        assert isinstance(s.lock, threading.Lock)


class TestSettingsGetAttr:
    """Test Settings __getattr__ method."""

    def test_getattr_from_main_config(self):
        """Test getting attribute from main_thread_config."""
        settings_module.main_thread_config["test_key"] = "test_value"
        s = Settings()
        assert s.test_key == "test_value"

    def test_getattr_from_thread_local_overrides(self):
        """Test that thread local overrides take precedence."""
        settings_module.main_thread_config["key"] = "main_value"
        settings_module.thread_local_overrides.set({"key": "override_value"})
        s = Settings()
        assert s.key == "override_value"

    def test_getattr_raises_attribute_error(self):
        """Test that accessing non-existent attribute raises AttributeError."""
        s = Settings()
        with pytest.raises(AttributeError, match="'Settings' object has no attribute 'nonexistent'"):
            _ = s.nonexistent

    def test_getattr_with_default_local_lm(self):
        """Test getting default local_lm attribute."""
        s = Settings()
        assert s.local_lm is None


class TestSettingsSetAttr:
    """Test Settings __setattr__ method."""

    def test_setattr_calls_configure(self):
        """Test that setting an attribute calls configure."""
        s = Settings()
        s.new_key = "new_value"
        assert settings_module.main_thread_config["new_key"] == "new_value"

    def test_setattr_instance_not_configured(self):
        """Test that _instance is handled specially."""
        # This is tested implicitly by singleton creation
        s = Settings()
        assert s._instance is not None


class TestSettingsItemAccess:
    """Test Settings __getitem__ and __setitem__."""

    def test_getitem(self):
        """Test dictionary-style access."""
        settings_module.main_thread_config["key"] = "value"
        s = Settings()
        assert s["key"] == "value"

    def test_setitem(self):
        """Test dictionary-style setting."""
        s = Settings()
        s["new_key"] = "new_value"
        assert settings_module.main_thread_config["new_key"] == "new_value"

    def test_getitem_raises_attribute_error(self):
        """Test that accessing non-existent key raises AttributeError."""
        s = Settings()
        with pytest.raises(AttributeError):
            _ = s["nonexistent"]


class TestSettingsContains:
    """Test Settings __contains__ method."""

    def test_contains_in_main_config(self):
        """Test that 'in' operator works with main_thread_config."""
        settings_module.main_thread_config["key"] = "value"
        s = Settings()
        assert "key" in s

    def test_contains_in_overrides(self):
        """Test that 'in' operator works with thread local overrides."""
        settings_module.thread_local_overrides.set({"override_key": "value"})
        s = Settings()
        assert "override_key" in s

    def test_not_contains(self):
        """Test that 'in' operator returns False for non-existent keys."""
        s = Settings()
        assert "nonexistent" not in s


class TestSettingsGet:
    """Test Settings get() method."""

    def test_get_existing_key(self):
        """Test get() returns value for existing key."""
        settings_module.main_thread_config["key"] = "value"
        s = Settings()
        assert s.get("key") == "value"

    def test_get_nonexistent_key_returns_none(self):
        """Test get() returns None for non-existent key."""
        s = Settings()
        assert s.get("nonexistent") is None

    def test_get_with_default(self):
        """Test get() returns default for non-existent key."""
        s = Settings()
        assert s.get("nonexistent", "default") == "default"

    def test_get_existing_key_ignores_default(self):
        """Test get() returns actual value even when default is provided."""
        settings_module.main_thread_config["key"] = "value"
        s = Settings()
        assert s.get("key", "default") == "value"


class TestSettingsCopy:
    """Test Settings copy() and config property."""

    def test_copy_returns_dict(self):
        """Test that copy() returns a dictionary."""
        s = Settings()
        config = s.copy()
        assert isinstance(config, dict)

    def test_copy_includes_main_config(self):
        """Test that copy() includes main_thread_config."""
        settings_module.main_thread_config["key"] = "value"
        s = Settings()
        config = s.copy()
        assert config["key"] == "value"

    def test_copy_includes_overrides(self):
        """Test that copy() includes thread local overrides."""
        settings_module.main_thread_config["key1"] = "value1"
        settings_module.thread_local_overrides.set({"key2": "value2"})
        s = Settings()
        config = s.copy()
        assert config["key1"] == "value1"
        assert config["key2"] == "value2"

    def test_copy_overrides_take_precedence(self):
        """Test that overrides take precedence in copy()."""
        settings_module.main_thread_config["key"] = "main_value"
        settings_module.thread_local_overrides.set({"key": "override_value"})
        s = Settings()
        config = s.copy()
        assert config["key"] == "override_value"

    def test_config_property(self):
        """Test that config property calls copy()."""
        settings_module.main_thread_config["key"] = "value"
        s = Settings()
        assert s.config == s.copy()


class TestSettingsConfigure:
    """Test Settings configure() method."""

    def test_configure_sets_main_config(self):
        """Test that configure() sets values in main_thread_config."""
        s = Settings()
        s.configure(key="value")
        assert settings_module.main_thread_config["key"] == "value"

    def test_configure_multiple_values(self):
        """Test configuring multiple values at once."""
        s = Settings()
        s.configure(key1="value1", key2="value2")
        assert settings_module.main_thread_config["key1"] == "value1"
        assert settings_module.main_thread_config["key2"] == "value2"

    def test_configure_sets_owner_thread(self):
        """Test that first configure() call sets owner thread."""
        s = Settings()
        s.configure(key="value")
        assert settings_module.config_owner_thread_id == threading.get_ident()

    def test_configure_from_different_thread_raises(self):
        """Test that configure from different thread raises RuntimeError."""
        s = Settings()
        s.configure(key="value")  # First call sets owner

        error_raised = threading.Event()
        exception_holder = []

        def other_thread():
            try:
                s.configure(other_key="other_value")
            except RuntimeError as e:
                exception_holder.append(e)
                error_raised.set()

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=1)

        assert error_raised.is_set()
        assert len(exception_holder) == 1
        assert "dspy.settings can only be changed by the thread that initially configured it" in str(
            exception_holder[0]
        )


class TestSettingsContext:
    """Test Settings context() context manager."""

    def test_context_sets_thread_local_override(self):
        """Test that context manager sets thread local overrides."""
        s = Settings()
        with s.context(key="override_value"):
            assert s.key == "override_value"

    def test_context_does_not_affect_main_config(self):
        """Test that context manager doesn't modify main_thread_config."""
        settings_module.main_thread_config["key"] = "main_value"
        s = Settings()
        with s.context(key="override_value"):
            assert s.key == "override_value"
        # After context, should be back to main config
        assert s.key == "main_value"

    def test_context_cleanup_on_exit(self):
        """Test that context manager cleans up on exit."""
        s = Settings()
        settings_module.main_thread_config["key"] = "main_value"
        with s.context(key="override_value"):
            pass
        assert s.key == "main_value"

    def test_context_multiple_overrides(self):
        """Test context manager with multiple overrides."""
        s = Settings()
        with s.context(key1="value1", key2="value2"):
            assert s.key1 == "value1"
            assert s.key2 == "value2"

    def test_context_preserves_existing_overrides(self):
        """Test that nested context preserves previous overrides."""
        s = Settings()
        settings_module.thread_local_overrides.set({"existing": "value"})
        with s.context(new_key="new_value"):
            assert s.existing == "value"
            assert s.new_key == "new_value"

    def test_context_cleanup_on_exception(self):
        """Test that context manager cleans up even on exception."""
        s = Settings()
        settings_module.main_thread_config["key"] = "main_value"

        with pytest.raises(ValueError, match="test error"):
            with s.context(key="override_value"):
                raise ValueError("test error")

        assert s.key == "main_value"


class TestSettingsRepr:
    """Test Settings __repr__ method."""

    def test_repr_returns_string(self):
        """Test that __repr__ returns a string."""
        s = Settings()
        assert isinstance(repr(s), str)

    def test_repr_includes_config(self):
        """Test that __repr__ includes configuration."""
        settings_module.main_thread_config["key"] = "value"
        s = Settings()
        repr_str = repr(s)
        assert "key" in repr_str
        assert "value" in repr_str

    def test_repr_includes_overrides(self):
        """Test that __repr__ includes thread local overrides."""
        settings_module.thread_local_overrides.set({"override_key": "override_value"})
        s = Settings()
        repr_str = repr(s)
        assert "override_key" in repr_str


class TestSettingsEnsureConfigureAllowed:
    """Test Settings _ensure_configure_allowed() method."""

    def test_ensure_configure_allowed_first_call(self):
        """Test that first call to _ensure_configure_allowed succeeds."""
        s = Settings()
        s._ensure_configure_allowed()  # Should not raise
        assert settings_module.config_owner_thread_id == threading.get_ident()

    def test_ensure_configure_allowed_same_thread(self):
        """Test that repeated calls from same thread succeed."""
        s = Settings()
        s._ensure_configure_allowed()
        s._ensure_configure_allowed()  # Should not raise

    @patch("asyncio.current_task")
    def test_ensure_configure_allowed_in_async_task(self, mock_current_task):
        """Test that first async task can call configure."""
        mock_task = Mock()
        mock_current_task.return_value = mock_task

        s = Settings()
        s._ensure_configure_allowed()  # Should not raise
        assert settings_module.config_owner_async_task is mock_task

    @patch("asyncio.current_task")
    def test_ensure_configure_allowed_different_async_task_raises(self, mock_current_task):
        """Test that different async task cannot call configure."""
        mock_task1 = Mock()
        mock_task2 = Mock()
        mock_current_task.return_value = mock_task1

        s = Settings()
        s._ensure_configure_allowed()  # First call sets owner

        mock_current_task.return_value = mock_task2
        with pytest.raises(RuntimeError, match="can only be called from the same async task"):
            s._ensure_configure_allowed()

    @patch("asyncio.current_task")
    def test_ensure_configure_allowed_no_async_task(self, mock_current_task):
        """Test handling when not in an async task."""
        mock_current_task.side_effect = RuntimeError("no current task")

        s = Settings()
        s._ensure_configure_allowed()  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
