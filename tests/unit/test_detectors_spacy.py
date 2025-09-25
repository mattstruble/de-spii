from unittest import mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from despii.core import DeSPII, reconstruct_text
from despii.detectors import spacy as spacy_mod


def make_mock_spacy(entities):
    class MockSpan:
        def __init__(self, text, label):
            self.text = text
            # spaCy's ent.label_ is a string; use str to ensure compatibility
            self.label_ = str(label)

    class MockDoc:
        def __init__(self, ents):
            self.ents = [MockSpan(t, l) for t, l in ents]

    class MockLanguage:
        def __call__(self, text):
            # return only spans that actually appear in the text
            present = [(t, l) for t, l in entities if t in text]
            return MockDoc(present)

    return MockLanguage()


@pytest.mark.parametrize(
    "label,text,entity",
    [
        (spacy_mod.Labels.PERSON, "Alice went home.", ("Alice", "PERSON")),
        (spacy_mod.Labels.ORG, "Work at OpenAI today.", ("OpenAI", "ORG")),
        (spacy_mod.Labels.GPE, "I love Paris in spring.", ("Paris", "GPE")),
        (spacy_mod.Labels.DATE, "Due on 2024-01-01.", ("2024-01-01", "DATE")),
        (spacy_mod.Labels.LOC, "Meet at Central Park.", ("Central Park", "LOC")),
    ],
)
def test_spacy_replaces_supported_labels(monkeypatch, label, text, entity):
    monkeypatch.setattr(spacy_mod, "spacy_model", lambda: make_mock_spacy([entity]))

    d = DeSPII(text=text, pii_map={})
    result = spacy_mod.spacy_pass(d)

    assert entity[0] not in result.text
    assert any(v == entity[0] for v in result.pii_map.values())

    reconstructed = reconstruct_text(result.text, result)
    assert reconstructed == text


def test_spacy_ignores_unsupported_labels(monkeypatch):
    entity = ("blue", "LANGUAGE")
    text = "The word blue appears."
    monkeypatch.setattr(spacy_mod, "spacy_model", lambda: make_mock_spacy([entity]))

    d = DeSPII(text=text, pii_map={})
    result = spacy_mod.spacy_pass(d)

    assert result.text == text
    assert result.pii_map == {}
    assert result.count == 0


@given(st.text(min_size=0, max_size=200))
def test_spacy_reconstruction_invariant(text):
    # No entities detected -> identity transform is still reconstructable
    with mock.patch.object(spacy_mod, "spacy_model", return_value=make_mock_spacy([])):
        d = DeSPII(text=text, pii_map={})
        result = spacy_mod.spacy_pass(d)
        assert reconstruct_text(result.text, result) == text


@pytest.mark.parametrize(
    "lang,expected",
    [
        ("en", "en_core_web_sm"),
        ("fr", "fr_core_news_sm"),
        ("xx", "xx_ent_wiki_sm"),
        (None, "xx_ent_wiki_sm"),
        ("zz", "xx_ent_wiki_sm"),
    ],
)
def test_spacy_model_for_lang_mapping(lang, expected):
    assert spacy_mod._spacy_model_for_lang(lang) == expected


def test_spacy_model_loads_selected_package(monkeypatch):
    # Force language and verify the selected package bubbles into loader
    monkeypatch.setattr(spacy_mod, "detect_system_lang", lambda: "en")

    captured = {}

    def fake_loader(name):
        captured["name"] = name
        return make_mock_spacy([])

    # monkeypatch loader and clear the spacy_model cache to ensure call-through
    monkeypatch.setattr(spacy_mod, "_load_spacy_model", fake_loader)
    spacy_mod.spacy_model.cache_clear()

    _ = spacy_mod.spacy_model()
    assert captured.get("name") == "en_core_web_sm"
