from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from despii.detectors import spacy as spacy_mod


def make_mock_spacy(entities):
    class MockSpan:
        def __init__(self, text, label):
            self.text = text
            # spaCy's ent.label_ is a string; use str to ensure compatibility
            self.label_ = str(label)

    class MockDoc:
        def __init__(self, ents):
            self.ents = [
                MockSpan(text_value, label_value) for text_value, label_value in ents
            ]

    class MockLanguage:
        def __call__(self, text):
            # return only spans that actually appear in the text
            present = [
                (text_value, label_value)
                for text_value, label_value in entities
                if text_value in text
            ]
            return MockDoc(present)

    return MockLanguage()


@pytest.mark.parametrize(
    ("label", "text", "entity"),
    [
        (spacy_mod.Labels.PERSON, "Alice went home.", ("Alice", "PERSON")),
        (spacy_mod.Labels.ORG, "Work at OpenAI today.", ("OpenAI", "ORG")),
        (spacy_mod.Labels.GPE, "I love Paris in spring.", ("Paris", "GPE")),
        (spacy_mod.Labels.DATE, "Due on 2024-01-01.", ("2024-01-01", "DATE")),
        (spacy_mod.Labels.LOC, "Meet at Central Park.", ("Central Park", "LOC")),
    ],
)
def test_spacy_replaces_supported_labels(ctx_factory, monkeypatch, label, text, entity):
    monkeypatch.setattr(spacy_mod, "spacy_model", lambda: make_mock_spacy([entity]))

    ctx = ctx_factory(text)

    _ = spacy_mod.spacy_pass(ctx)
    ctx.redact.assert_any_call(entity[0], entity[1])


def test_spacy_ignores_unsupported_labels(ctx_factory, monkeypatch):
    entity = ("blue", "LANGUAGE")
    text = "The word blue appears."
    monkeypatch.setattr(spacy_mod, "spacy_model", lambda: make_mock_spacy([entity]))

    ctx = ctx_factory(text)

    _ = spacy_mod.spacy_pass(ctx)
    ctx.redact.assert_not_called()


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.text(min_size=0, max_size=200))
def test_spacy_no_entities_calls_no_redact(ctx_factory, text):
    # No entities detected -> redact should not be called
    with mock.patch.object(spacy_mod, "spacy_model", return_value=make_mock_spacy([])):
        ctx = ctx_factory(text)
        _ = spacy_mod.spacy_pass(ctx)
        ctx.redact.assert_not_called()


@pytest.mark.parametrize(
    ("lang", "expected"),
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


def test_spacy_download_fallback_on_missing_model(monkeypatch):
    # Ensure we exercise the try/except path in _load_spacy_model
    spacy_mod._load_spacy_model.cache_clear()

    sentinel_model = object()

    # First call raises, second returns a model object
    load_mock = mock.Mock()
    load_mock.side_effect = [Exception("missing"), sentinel_model]
    monkeypatch.setattr(spacy_mod.spacy, "load", load_mock)

    check_call_mock = mock.Mock(return_value=0)
    monkeypatch.setattr(spacy_mod.subprocess, "check_call", check_call_mock)

    model = spacy_mod._load_spacy_model("en_core_web_sm")
    assert model is sentinel_model

    # Verify we attempted to download exactly once
    assert check_call_mock.call_count == 1
    args, _ = check_call_mock.call_args
    cmd = args[0]
    assert "spacy" in cmd
    assert "download" in cmd
    assert "en_core_web_sm" in cmd
