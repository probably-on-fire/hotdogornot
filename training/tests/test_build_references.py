import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _make_tiny_dataset(root: Path, class_names: list[str], per_class: int = 3) -> None:
    for name in class_names:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(hash(name) % (2**32))
        for i in range(per_class):
            arr = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")


def test_build_references_writes_expected_binary(tmp_path: Path, classes_yaml: Path):
    from rfconnectorai.inference.build_references import build_references
    from rfconnectorai.models.embedder import RGBDEmbedder

    data_root = tmp_path / "refs"
    _make_tiny_dataset(data_root, ["SMA-M", "SMA-F"], per_class=3)

    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    ckpt = tmp_path / "embedder.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": 128}, ckpt)

    out_bin = tmp_path / "reference_embeddings.bin"
    build_references(
        checkpoint=ckpt,
        data_root=data_root,
        classes_yaml=classes_yaml,
        output_path=out_bin,
        image_size=64,
        device="cpu",
    )

    assert out_bin.exists()

    with open(out_bin, "rb") as f:
        magic = f.read(4)
        assert magic == b"RFCE"                       # RF Connector Embeddings
        version, n_classes, dim = struct.unpack("<III", f.read(12))
        assert version == 1
        assert n_classes == 2
        assert dim == 128

        # n_classes × (int32 id, 64 bytes name, dim × float32)
        for _ in range(n_classes):
            class_id = struct.unpack("<i", f.read(4))[0]
            name_bytes = f.read(64)
            name = name_bytes.split(b"\x00", 1)[0].decode("utf-8")
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            assert class_id in (0, 1)
            assert name in ("SMA-M", "SMA-F")
            # Vectors are L2-normalized.
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-5


def test_build_references_rejects_missing_class_dir(
    tmp_path: Path, classes_yaml: Path
):
    from rfconnectorai.inference.build_references import build_references
    from rfconnectorai.models.embedder import RGBDEmbedder

    data_root = tmp_path / "refs"
    _make_tiny_dataset(data_root, ["SMA-M"], per_class=2)  # missing SMA-F

    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    ckpt = tmp_path / "e.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": 128}, ckpt)

    out_bin = tmp_path / "r.bin"
    with pytest.raises(FileNotFoundError):
        build_references(
            checkpoint=ckpt,
            data_root=data_root,
            classes_yaml=classes_yaml,
            output_path=out_bin,
            image_size=64,
            device="cpu",
        )
