from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rfconnectorai.data.classes import load_classes
from rfconnectorai.data.dataset import RGBDConnectorDataset
from rfconnectorai.models.embedder import RGBDEmbedder


def _load_references(path: Path) -> tuple[list[int], list[str], np.ndarray]:
    with open(path, "rb") as f:
        if f.read(4) != b"RFCE":
            raise ValueError("bad magic in references file")
        version, n_classes, dim = struct.unpack("<III", f.read(12))
        if version != 1:
            raise ValueError(f"unsupported references version {version}")
        ids: list[int] = []
        names: list[str] = []
        vectors = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            (cid,) = struct.unpack("<i", f.read(4))
            name = f.read(64).split(b"\x00", 1)[0].decode("utf-8")
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            ids.append(cid)
            names.append(name)
            vectors[i] = vec
    return ids, names, vectors


def _expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    ece = 0.0
    total = len(confidences)
    if total == 0:
        return 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        acc_bin = correct[in_bin].mean()
        conf_bin = confidences[in_bin].mean()
        ece += (in_bin.sum() / total) * abs(acc_bin - conf_bin)
    return float(ece)


def evaluate(
    checkpoint: Path,
    references: Path,
    data_root: Path,
    classes_yaml: Path,
    output_path: Path,
    image_size: int = 384,
    device: str = "cpu",
) -> dict:
    classes = load_classes(classes_yaml)
    id_to_name = {c.id: c.name for c in classes}

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    embedding_dim = ckpt["embedding_dim"]
    backbone = ckpt.get("backbone", "mobilevitv2_100")
    model = RGBDEmbedder(embedding_dim=embedding_dim, pretrained=False, backbone=backbone).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ref_ids, ref_names, ref_vectors = _load_references(references)
    ref_t = torch.from_numpy(ref_vectors).to(device)

    ds = RGBDConnectorDataset(
        root=data_root, classes_yaml=classes_yaml, image_size=image_size
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    all_pred: list[int] = []
    all_true: list[int] = []
    all_conf: list[float] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model(x)
            sims = emb @ ref_t.T           # (B, n_classes)   cosine since all L2-normalized
            top_sim, top_idx = sims.max(dim=1)
            pred = torch.tensor([ref_ids[i.item()] for i in top_idx])
            # Convert cosine sim [-1, 1] to pseudo-probability [0, 1] via linear scaling.
            conf = (top_sim.cpu().numpy() + 1.0) / 2.0

            all_pred.extend(pred.tolist())
            all_true.extend(y.tolist())
            all_conf.extend(conf.tolist())

    pred_arr = np.array(all_pred)
    true_arr = np.array(all_true)
    conf_arr = np.array(all_conf)
    correct = (pred_arr == true_arr).astype(np.int32)

    top1 = float(correct.mean())

    per_class_recall: dict[str, float] = {}
    for c in classes:
        mask = true_arr == c.id
        if mask.sum() == 0:
            per_class_recall[c.name] = 0.0
        else:
            per_class_recall[c.name] = float(correct[mask].mean())

    n = len(classes)
    confusion = np.zeros((n, n), dtype=int)
    for t, p in zip(true_arr, pred_arr):
        confusion[t][p] += 1

    ece = _expected_calibration_error(conf_arr, correct.astype(float))

    report = {
        "top1_accuracy": top1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": [id_to_name[c.id] for c in classes],
        "expected_calibration_error": ece,
        "n_samples": int(len(true_arr)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--references", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--classes-yaml", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    report = evaluate(
        checkpoint=args.checkpoint,
        references=args.references,
        data_root=args.data_root,
        classes_yaml=args.classes_yaml,
        output_path=args.output,
        image_size=args.image_size,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
