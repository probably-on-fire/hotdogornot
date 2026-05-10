from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rfconnectorai.classifier.label_encoding import (
    HEAD_ORDER,
    HeadVocab,
    MISSING_INDEX,
    class_balance,
    default_head_vocabs,
    encode_attributes,
    encode_row,
    read_attributes_csv,
)


def test_head_vocab_requires_unknown():
    with pytest.raises(ValueError, match="unknown"):
        HeadVocab("family", ("SMA", "BNC"))


def test_head_vocab_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicates"):
        HeadVocab("family", ("SMA", "SMA", "unknown"))


def test_head_vocab_index_round_trip():
    vocab = HeadVocab("family", ("SMA", "BNC", "unknown"))
    assert vocab.num_classes == 3
    assert vocab.index_of("SMA") == 0
    assert vocab.index_of("BNC") == 1
    assert vocab.index_of("unknown") == 2
    assert vocab.value_of(0) == "SMA"
    assert vocab.value_of(MISSING_INDEX) == "<missing>"


def test_missing_value_returns_missing_index():
    vocab = HeadVocab("polarity", ("standard", "not_applicable", "unknown"))
    assert vocab.index_of(None) == MISSING_INDEX
    assert vocab.index_of("") == MISSING_INDEX
    assert vocab.index_of("garbage") == MISSING_INDEX


def test_default_head_vocabs_covers_head_order():
    vocabs = default_head_vocabs(["SMA", "BNC"])
    for head in HEAD_ORDER:
        assert head in vocabs
        assert "unknown" in vocabs[head].values


def test_encode_row_handles_missing_fields():
    vocabs = default_head_vocabs(["SMA"])
    encoded = encode_row(
        {
            "instance_id": "i1",
            "crop_path": "c1.jpg",
            "family": "SMA",
            "polarity": "standard",
        },
        vocabs,
    )
    assert encoded.targets["family"] == vocabs["family"].index_of("SMA")
    assert encoded.targets["polarity"] == vocabs["polarity"].index_of("standard")
    # Heads not in row should be MISSING_INDEX.
    assert encoded.targets["mount_style"] == MISSING_INDEX
    assert encoded.targets["finish_material_cue"] == MISSING_INDEX


def test_target_vector_uses_head_order():
    vocabs = default_head_vocabs(["SMA"])
    row = {"family": "SMA"}
    encoded = encode_row(row, vocabs)
    vec = encoded.target_vector()
    assert len(vec) == len(HEAD_ORDER)
    assert vec[HEAD_ORDER.index("family")] == vocabs["family"].index_of("SMA")
    assert vec[HEAD_ORDER.index("polarity")] == MISSING_INDEX


def test_read_attributes_csv_round_trip(tmp_path: Path):
    p = tmp_path / "attributes.csv"
    fields = ["instance_id", "split", "family", "polarity"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"instance_id": "i1", "split": "train", "family": "SMA", "polarity": "standard"})
        writer.writerow({"instance_id": "i2", "split": "val", "family": "BNC", "polarity": "not_applicable"})
    rows = read_attributes_csv(p)
    assert len(rows) == 2
    assert rows[0]["family"] == "SMA"


def test_class_balance_counts_missing(tmp_path: Path):
    vocabs = default_head_vocabs(["SMA", "BNC"])
    rows = [
        {"instance_id": "a", "family": "SMA", "polarity": "standard"},
        {"instance_id": "b", "family": "BNC"},
        {"instance_id": "c", "family": "SMA"},
    ]
    encoded = encode_attributes(rows, vocabs)
    balance = class_balance(encoded, "polarity", vocabs["polarity"])
    assert balance["<missing>"] == 2
    assert balance["standard"] == 1
