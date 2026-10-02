"""YAML 1.2 scalars stay consistent between files and CLI overrides."""

from orch.yamlio import dump_text, load_text, parse_value


def test_scientific_notation_bool_and_null():
    assert parse_value("5e-5") == 5e-5
    assert isinstance(parse_value("5e-5"), float)
    assert parse_value("5.0e-5") == parse_value("5e-5")
    assert isinstance(parse_value("2"), int)
    assert isinstance(parse_value("2.0"), float)
    assert parse_value("true") is True
    assert parse_value("null") is None
    assert parse_value("[a, b]") == ["a", "b"]
    assert parse_value("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"


def test_plain_strings_that_yaml_11_would_coerce():
    loaded = load_text("mem: 24G\nflag: yes\n")
    assert loaded["mem"] == "24G"
    assert loaded["flag"] == "yes"


def test_dump_round_trip_preserves_types():
    payload = {"lr": 5e-5, "name": "Qwen/Qwen2.5-1.5B-Instruct", "clip": None, "ok": True}
    again = load_text(dump_text(payload))
    assert again["lr"] == payload["lr"]
    assert isinstance(again["lr"], float)
    assert again["clip"] is None
    assert again["ok"] is True
