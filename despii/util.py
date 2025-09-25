import locale
import os
from enum import Enum
from functools import lru_cache
from typing import Optional, override


class StrEnum(str, Enum):
    """Allow inheriting enums to act as strings."""

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):  # noqa: ANN001, ANN205
        """Set `auto()` to use the enum variable name cast to upper."""
        return name

    @override
    def __str__(self) -> str:
        """Return enum value."""
        return self.value


@lru_cache(maxsize=1)
def detect_system_lang() -> Optional[str]:
    """Detect system language from $LANG (if set) or locale.getdefaultlocale().

    :return: a two-letter language code (e.g., 'en', 'fr') or None if unknown.
    """
    lang = os.environ.get("LANG")
    if lang:
        code = lang.split(".")[0].split("_")[0]
        if len(code) >= 2:
            return code[:2].lower()

    loc = locale.getdefaultlocale()[0]
    if loc:
        code = str(loc).split("_")[0]
        if len(code) >= 2:
            return code[:2].lower()

    return None
