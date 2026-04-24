from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest


def test_pretrained_export_produces_loadable_onnx(tmp_path: Path):
    from rfconnectorai.export.make_pretrained_embedder import export

    out = tmp_path / "pretrained.onnx"
    # mobilevit is faster to load than dinov2 for tests
    export(output=out, backbone="mobilevitv2_100", image_size=64, quantize=False)

    assert out.exists()
    assert out.stat().st_size > 1024  # not empty

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 4, 64, 64).astype(np.float32)
    out_arr = session.run(None, {"input": x})[0]
    assert out_arr.shape == (1, 128)
    norms = np.linalg.norm(out_arr, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)


def test_pretrained_export_with_quantize(tmp_path: Path):
    from rfconnectorai.export.make_pretrained_embedder import export

    out = tmp_path / "pretrained.onnx"
    export(output=out, backbone="mobilevitv2_100", image_size=64, quantize=True)

    int8 = tmp_path / "pretrained_int8.onnx"
    assert out.exists()
    assert int8.exists()
    assert int8.stat().st_size < out.stat().st_size * 0.6  # meaningful shrink
