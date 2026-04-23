from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rfconnectorai.data.classes import load_classes
from rfconnectorai.data.dataset import RGBDConnectorDataset
from rfconnectorai.models.embedder import RGBDEmbedder


MAGIC = b"RFCE"
FORMAT_VERSION = 1


def build_references(
    checkpoint: Path,
    data_root: Path,
    classes_yaml: Path,
    output_path: Path,
    image_size: int = 384,
    device: str = "cpu",
) -> Path:
    classes = load_classes(classes_yaml)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    embedding_dim = ckpt["embedding_dim"]
    model = RGBDEmbedder(embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = RGBDConnectorDataset(
        root=data_root, classes_yaml=classes_yaml, image_size=image_size
    )

    # Accumulate per-class embedding sums.
    sums: dict[int, torch.Tensor] = {c.id: torch.zeros(embedding_dim) for c in classes}
    counts: dict[int, int] = {c.id: 0 for c in classes}

    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model(x).cpu()
            for i, lab in enumerate(y.tolist()):
                sums[lab] += emb[i]
                counts[lab] += 1

    # Write binary file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", FORMAT_VERSION, len(classes), embedding_dim))
        for c in classes:
            if counts[c.id] == 0:
                raise RuntimeError(f"No samples found for class {c.name}")
            mean = sums[c.id] / counts[c.id]
            mean = F.normalize(mean.unsqueeze(0), p=2, dim=1).squeeze(0)
            f.write(struct.pack("<i", c.id))
            name_bytes = c.name.encode("utf-8")
            if len(name_bytes) > 64:
                raise ValueError(f"Class name too long: {c.name}")
            f.write(name_bytes.ljust(64, b"\x00"))
            f.write(mean.numpy().astype("float32").tobytes())

    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--classes-yaml", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    out = build_references(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        classes_yaml=args.classes_yaml,
        output_path=args.output,
        image_size=args.image_size,
        device=args.device,
    )
    print(f"reference embeddings written to {out}")


if __name__ == "__main__":
    main()
