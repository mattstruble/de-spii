import subprocess  # nosec B404
import sys
from enum import auto
from functools import lru_cache

import spacy

from despii.core import DeSPII
from despii.util import StrEnum, detect_system_lang


class Labels(StrEnum):
    """Spacy ner labels.

    Generated using:  https://stackoverflow.com/a/78475321

    nlp = spacy.load(
        "en_core_web_trf",
        disable=["tagger", "parser", "attribute_ruler", "lemmatizer"]
    )
    for label in nlp.get_pipe('ner').labels:
        print(f"{label}: {spacy.explain(label)}")
    """

    CARDINAL = auto()  # Numerals that do not fall under another type
    DATE = auto()  # Absolute or relative dates or periods
    EVENT = auto()  # Named hurricanes, battles, wars, sports events, etc.
    FAC = auto()  # Buildings, airports, highways, bridges, etc.
    GPE = auto()  # Countries, cities, states
    LANGUAGE = auto()  # Any named language
    LAW = auto()  # Named documents made into laws.
    LOC = auto()  # Non-GPE locations, mountain ranges, bodies of water
    MONEY = auto()  # Monetary values, including unit
    NORP = auto()  # Nationalities or religious or political groups
    ORDINAL = auto()  # "first", "second", etc.
    ORG = auto()  # Companies, agencies, institutions, etc.
    PERCENT = auto()  # Percentage, including "%"
    PERSON = auto()  # People, including fictional
    PRODUCT = auto()  # Objects, vehicles, foods, etc. (not services)
    QUANTITY = auto()  # Measurements, as of weight or distance
    TIME = auto()  # Times smaller than a day
    WORK_OF_ART = auto()  # Titles of books, songs, etc.


_PII_LABELS = {
    Labels.PERSON,
    Labels.ORG,
    Labels.GPE,
    Labels.DATE,
    Labels.LOC,
}


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


def spacy_pass(despii: DeSPII) -> DeSPII:
    nlp = spacy_model()
    doc = nlp(despii.text)
    for ent in doc.ents:
        label = ent.label_
        print(label, _PII_LABELS)
        if label in _PII_LABELS:
            if ent.text not in despii.pii_map.values():
                placeholder = despii.create_placeholder(label)
                despii.pii_map[placeholder] = ent.text
                despii.text = despii.text.replace(ent.text, placeholder)
    return despii
