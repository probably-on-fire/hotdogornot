from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from rfconnectorai.export.onnx_export import export_embedder, export_detector
from rfconnectorai.models.embedder import RGBDEmbedder


def test_export_embedder_roundtrip(tmp_path: Path):
    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    model.eval()
    ckpt = tmp_path / "e.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": 128}, ckpt)

    onnx_path = tmp_path / "embedder.onnx"
    export_embedder(checkpoint=ckpt, output=onnx_path, image_size=64)

    assert onnx_path.exists()

    # Compare PyTorch vs ONNX Runtime outputs.
    x = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        torch_out = model(x).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"input": x.numpy()})[0]

    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-4)


def test_export_detector_produces_onnx(tmp_path: Path):
    """
    The detector export is an Ultralytics passthrough — we just verify it yields
    an ONNX file. Deeper equivalence checks happen inside ultralytics.
    """
    onnx_path = tmp_path / "detector.onnx"

    export_detector(weights=Path("yolo11n.pt"), output=onnx_path, image_size=320)

    assert onnx_path.exists()
    # Load with onnxruntime to ensure it's valid.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session is not None
