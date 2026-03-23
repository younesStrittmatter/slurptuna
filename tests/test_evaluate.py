from __future__ import annotations

from slurptuna.evaluate import normalize_seed_loss, summarize_rows


def test_normalize_scalar():
    row = normalize_seed_loss(3, 1.25)
    assert row["seed"] == 3
    assert row["total_loss"] == 1.25


def test_normalize_dict_components():
    row = normalize_seed_loss(2, {"a": 2.0, "b": 4.0})
    assert row["total_loss"] == 3.0
    assert row["components"] == {"a": 2.0, "b": 4.0}


def test_summarize_rows():
    summary = summarize_rows(
        [
            {"seed": 0, "total_loss": 1.0, "components": {}},
            {"seed": 1, "total_loss": 3.0, "components": {}},
        ]
    )
    assert summary["n_seeds"] == 2
    assert summary["mean_total_loss"] == 2.0
