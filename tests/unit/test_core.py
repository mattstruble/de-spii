import re

from despii.core import RedactionContext


def test_redactioncontext_redact_and_unredact_single():
    ctx = RedactionContext("Call me at 555-123-4567")
    ctx.redact("555-123-4567", "phone")

    assert "555-123-4567" not in ctx.text
    assert re.search(r"<PII_PHONE_\d+>", ctx.text)
    assert ctx.unredact(ctx.text) == "Call me at 555-123-4567"


def test_redactioncontext_avoids_duplicate_entries():
    text = "Email me at test@example.com or test@example.com"
    ctx = RedactionContext(text)
    ctx.redact("test@example.com", "email")
    ctx.redact("test@example.com", "email")

    placeholders = re.findall(r"<PII_EMAIL_\d+>", ctx.text)
    # Only one unique placeholder should exist, but appear twice
    assert len(set(placeholders)) == 1
    assert ctx.text.count(list(set(placeholders))[0]) == 2
    assert ctx.unredact(ctx.text) == text


def test_redactioncontext_separate_counters_per_label():
    ctx = RedactionContext("Alice works at OpenAI")
    ctx.redact("Alice", "PERSON")
    ctx.redact("OpenAI", "ORG")

    assert re.search(r"<PII_PERSON_1>", ctx.text)
    assert re.search(r"<PII_ORG_1>", ctx.text)
