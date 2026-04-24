from pathlib import Path

import pytest
import torch
from rfconnectorai.data.dataset import RGBDConnectorDataset


def test_dataset_length_matches_image_count(synthetic_image_dir: Path, classes_yaml: Path):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
    )
    assert len(ds) == 6  # 2 classes × 3 images


def test_dataset_item_is_rgbd_tensor_and_label(
    synthetic_image_dir: Path, classes_yaml: Path
):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
    )
    rgbd, label = ds[0]
    assert isinstance(rgbd, torch.Tensor)
    assert rgbd.shape == (4, 64, 64)
    assert rgbd.dtype == torch.float32
    assert isinstance(label, int)
    assert label in (0, 1)


def test_dataset_rgb_channels_normalized(synthetic_image_dir: Path, classes_yaml: Path):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
    )
    rgbd, _ = ds[0]
    rgb = rgbd[:3]
    # After ImageNet normalization, typical values land in ~[-2.5, 2.5]
    assert rgb.min() > -3.0
    assert rgb.max() < 3.0


def test_dataset_depth_channel_in_meters(synthetic_image_dir: Path, classes_yaml: Path):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
    )
    rgbd, _ = ds[0]
    depth = rgbd[3]
    # Depth in the clamped range of synthesize_depth_from_mask
    assert depth.min() >= 0.0
    assert depth.max() < 5.0


def test_dataset_labels_come_from_classes_yaml(
    synthetic_image_dir: Path, classes_yaml: Path
):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
    )
    labels = {ds[i][1] for i in range(len(ds))}
    assert labels == {0, 1}


def test_dataset_missing_class_directory_raises(tmp_path: Path, classes_yaml: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        RGBDConnectorDataset(root=empty, classes_yaml=classes_yaml, image_size=64)


def test_dataset_with_augment_yields_different_outputs_per_call(
    synthetic_image_dir: Path, classes_yaml: Path
):
    """
    Augmented dataset should produce different RGB tensors across __getitem__
    calls for the same index (rotation, color jitter etc. are stochastic).
    """
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
        augment=True,
    )
    a, _ = ds[0]
    b, _ = ds[0]
    # RGB channels (0..2) should differ; depth (3) is deterministic per idx.
    assert not torch.allclose(a[:3], b[:3])


def test_dataset_without_augment_is_deterministic(
    synthetic_image_dir: Path, classes_yaml: Path
):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
        augment=False,
    )
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.allclose(a, b)
