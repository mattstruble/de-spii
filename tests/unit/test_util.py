from enum import auto
from unittest import mock

from despii.util import StrEnum, detect_system_lang


class Color(StrEnum):
    RED = auto()
    BLUE = auto()


def test_strenum_auto_value_and_str():
    assert Color.RED.value == "RED"
    assert str(Color.BLUE) == "BLUE"
    assert Color.RED == "RED"


def test_detect_system_lang_from_env(monkeypatch):
    detect_system_lang.cache_clear()
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert detect_system_lang() == "en"


def test_detect_system_lang_from_locale(monkeypatch):
    detect_system_lang.cache_clear()
    monkeypatch.delenv("LANG", raising=False)
    with mock.patch("locale.getdefaultlocale", return_value=("fr_FR", "UTF-8")):
        assert detect_system_lang() == "fr"


def test_detect_system_lang_none(monkeypatch):
    detect_system_lang.cache_clear()
    monkeypatch.delenv("LANG", raising=False)
    with mock.patch("locale.getdefaultlocale", return_value=(None, None)):
        assert detect_system_lang() is None
