import subprocess  # nosec B404
import sys
from functools import lru_cache

import spacy

from despii.util import detect_system_lang


@lru_cache(maxsize=1)
def _load_spacy_model(model_name: str) -> spacy.language.Language:
    try:
        return spacy.load(model_name)
    except Exception:
        _ = subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name]  # nosec B603
        )
        return spacy.load(model_name)


def _spacy_model_for_lang(lang_code: str | None) -> str:
    """Map a two-letter lang code to a spaCy core_web_sm model package name.

    Defaults to 'xx_ent_wiki_sm' for unknown or unsupported languages.

    :param lang_code: two-letter language code
    :return: supported spaCy model
    """
    mapping = {
        "ca": "ca_core_news_sm",
        "da": "da_core_news_sm",
        "de": "de_core_news_sm",
        "el": "el_core_news_sm",
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
        "fi": "fi_core_news_sm",
        "fr": "fr_core_news_sm",
        "hr": "hr_core_news_sm",
        "it": "it_core_news_sm",
        "ja": "ja_core_news_sm",
        "ko": "ko_core_news_sm",
        "lt": "lt_core_news_sm",
        "mk": "mk_core_news_sm",
        "nb": "nb_core_news_sm",
        "nl": "nl_core_news_sm",
        "pt": "pt_core_news_sm",
        "ro": "ro_core_news_sm",
        "ru": "ru_core_news_sm",
        "sl": "sl_core_news_sm",
        "sv": "sv_core_news_sm",
        "uk": "uk_core_news_sm",
        "xx": "xx_ent_wiki_sm",  # multi language
        "zh": "zh_core_web_sm",
    }

    return mapping.get(lang_code, mapping["xx"])


@lru_cache(maxsize=1)
def spacy_model() -> spacy.language.Language:
    lang = detect_system_lang()
    model_name = _spacy_model_for_lang(lang)
    return _load_spacy_model(model_name)
