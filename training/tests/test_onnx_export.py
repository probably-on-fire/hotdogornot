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


def test_quantize_int8_shrinks_model_and_preserves_outputs(tmp_path: Path):
    from rfconnectorai.export.onnx_export import quantize_int8

    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    model.eval()
    ckpt = tmp_path / "e.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": 128}, ckpt)

    onnx_path = tmp_path / "embedder.onnx"
    export_embedder(checkpoint=ckpt, output=onnx_path, image_size=64)

    int8_path = tmp_path / "embedder_int8.onnx"
    quantize_int8(onnx_path, int8_path)

    assert int8_path.exists()
    # Quantized model should be meaningfully smaller (typically 2-4x).
    fp_size = onnx_path.stat().st_size
    q_size = int8_path.stat().st_size
    assert q_size < fp_size * 0.6, f"INT8 model {q_size} not smaller than 60% of FP {fp_size}"

    # Outputs should be close to the float model on a fresh input.
    x = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        torch_out = model(x).numpy()

    fp_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    fp_out = fp_session.run(None, {"input": x.numpy()})[0]

    q_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    q_out = q_session.run(None, {"input": x.numpy()})[0]

    # FP and INT8 should both be close to the PyTorch output. Tolerance on
    # INT8 is looser because of quantization rounding.
    np.testing.assert_allclose(torch_out, fp_out, rtol=1e-3, atol=1e-4)
    # INT8 vs FP cosine similarity should remain very high even if absolute
    # values shift slightly. This is the property the downstream matcher
    # actually cares about.
    cos = float((fp_out * q_out).sum() / (
        (fp_out ** 2).sum() ** 0.5 * (q_out ** 2).sum() ** 0.5
    ))
    assert cos > 0.95, f"INT8 vs FP cosine similarity {cos:.4f} too low"


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
