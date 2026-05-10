# Mobile / Server Exports

This directory holds exported model artifacts produced by
``training/rfconnectorai/export/export_mobile.py`` and a manifest tying
each artifact back to its training run.

## Layout

```text
exports/mobile/
  exports_manifest.json
  detector_<arch>_<model_id>.onnx
  detector_<arch>_<model_id>.tflite
  classifier_<arch>_<model_id>.onnx
  classifier_<arch>_<model_id>.coreml
```

Every artifact filename embeds the ``model_id`` from
``training/rfconnectorai/models/registry.py`` so it can be matched back
to the exact training run, dataset hash, and taxonomy hash.

## Producing exports

Local PCs only run ``--dry-run`` (no torch/onnx/coremltools install
required). Real exports run in Kaggle / Colab where the ML toolchain is
available.

```bash
python -m rfconnectorai.export.export_mobile \
    --target detector:models/detector/best.pt:reports/experiments/<run>/model_record.json:onnx,tflite \
    --target classifier:models/multihead_classifier/best.pt:reports/experiments/<run>/model_record.json:onnx,coreml \
    --out exports/mobile \
    --dry-run
```

After a real cloud run, the manifest at
``exports/mobile/exports_manifest.json`` lists every exported artifact
with:

- target name (``detector`` / ``classifier``),
- output format,
- source artifact path,
- ``model_record.json`` reference,
- model_id, architecture, dataset id, and taxonomy hash.

## Compatibility Notes

- ONNX is always produced. ONNX Runtime is the most portable option for
  both server and mobile.
- TFLite/LiteRT requires a ``tensorflow`` or ``ai-edge-torch`` install in
  the cloud env.
- Core ML requires ``coremltools`` and is macOS-friendliest.
- Mobile latency / thermal benchmarks live in the per-device benchmark
  reports under ``reports/experiments/<run>/latency_report.md``.

## Hard rule

Mobile exports must not silently drop heads, change attribute vocabularies,
or break the legacy ``/predict`` response. Every exported artifact must
match the head vocabulary recorded in
``reports/experiments/<run>/head_vocabs.json``.
