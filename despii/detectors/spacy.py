import subprocess
import sys
from functools import lru_cache

import spacy


@lru_cache(maxsize=1)
def _load_spacy_model(model_name: str) -> spacy.language.Language:
    try:
        return spacy.load(model_name)
    except Exception:
        _ = subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name]
        )
        return spacy.load(model_name)


def _spacy_model_for_lang(lang_code: str | None) -> str:
    """Map a two-letter lang code to a spaCy core_web_sm model package name.

    Defaults to 'en_core_web_sm' for unknown or unsupported languages.

    :param lang_code: two-letter language code
    :return: supported spaCy model
    """
    lang_code = lang_code if lang_code else "en"

    mapping = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "es": "es_core_news_sm",
        "pt": "pt_core_news_sm",
        "it": "it_core_news_sm",
        "nl": "nl_core_news_sm",
        "xx": "xx_ent_wiki_sm",
        "zh": "zh_core_web_sm",
        "ja": "ja_core_news_sm",
        "ru": "ru_core_news_sm",
    }

    return mapping.get(lang_code, mapping["en"])
