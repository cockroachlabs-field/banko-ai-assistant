"""The demo's editable content: categories, personas and seed profiles."""

from decimal import Decimal

import pytest

from banko_ai.config import catalog as C
from banko_ai.config.settings import reset_config


@pytest.fixture(autouse=True)
def _fresh():
    C.reset_catalog()
    yield
    C.reset_catalog()


def test_packaged_catalog_loads_and_cross_checks():
    cat = C.get_catalog()
    assert {p.username for p in cat.personas} == {"maya", "sam", "riley"}
    assert {p.spending_style for p in cat.personas} <= set(cat.profiles)
    assert {"diner", "subscriber", "saver", "balanced"} <= set(cat.profiles)
    for profile in cat.profiles.values():
        assert any(a.kind == "checking" for a in profile.accounts)
        for spec in profile.purchases:
            assert cat.category_for(spec.category).label == spec.category
        for bill in profile.bills:
            assert cat.category_for(bill.category).label == bill.category


def test_persona_ids_are_the_coach_defaults():
    ids = {p.username: p.user_id for p in C.get_catalog().personas}
    assert ids["maya"] == "00000000-0000-0000-0000-0000000000a1"
    assert ids["sam"] == "00000000-0000-0000-0000-0000000000a2"
    assert ids["riley"] == "00000000-0000-0000-0000-0000000000a3"


def test_category_lookup_is_case_insensitive_with_aliases_and_fallback():
    cat = C.get_catalog()
    assert cat.category_for("groceries").label == "Groceries"
    assert cat.category_for("FOOD").label == "Restaurant"
    assert cat.category_for("restaurant").key == "restaurant"
    assert cat.category_for("something unheard of").key == "other"
    assert cat.category_for(None).key == "other"


def test_money_fields_are_exact_decimals():
    profile = C.get_catalog().profiles["subscriber"]
    netflix = next(b for b in profile.bills if b.merchant == "Netflix")
    assert netflix.amount == Decimal("15.99")
    assert netflix.price_change.to == Decimal("22.99")


def test_override_file_replaces_personas(tmp_path, monkeypatch):
    f = tmp_path / "personas.yaml"
    f.write_text(
        "personas:\n"
        "  - user_id: 00000000-0000-0000-0000-0000000000b1\n"
        "    username: jordan\n"
        "    display_name: Jordan Lee\n"
        "    avatar_color: '#123456'\n"
        "    spending_style: balanced\n",
        encoding="utf-8")
    monkeypatch.setenv("PERSONAS_FILE", str(f))
    reset_config()
    C.reset_catalog()
    assert [p.username for p in C.get_catalog().personas] == ["jordan"]


def test_unknown_style_in_personas_is_rejected(tmp_path, monkeypatch):
    f = tmp_path / "personas.yaml"
    f.write_text(
        "personas:\n"
        "  - user_id: 00000000-0000-0000-0000-0000000000b2\n"
        "    username: ghost\n"
        "    display_name: Ghost\n"
        "    avatar_color: '#000000'\n"
        "    spending_style: nonexistent\n",
        encoding="utf-8")
    monkeypatch.setenv("PERSONAS_FILE", str(f))
    reset_config()
    C.reset_catalog()
    with pytest.raises(ValueError, match="nonexistent"):
        C.get_catalog()


def test_profile_referencing_a_missing_account_is_rejected():
    with pytest.raises(ValueError, match="no account 'nowhere'"):
        C.SeedProfile.model_validate({
            "style": "x",
            "accounts": [{"key": "checking", "kind": "checking", "name": "C", "institution": "B"}],
            "income": {"account": "nowhere", "amount": 100},
        })
