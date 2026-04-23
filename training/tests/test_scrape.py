from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rfconnectorai.data.scrape import (
    CatalogImage,
    save_catalog_image,
    sanitize_filename,
)


def test_sanitize_filename_removes_unsafe_chars():
    assert sanitize_filename("sma/m connector??.jpg") == "sma_m_connector__.jpg"
    assert sanitize_filename("normal.png") == "normal.png"


def test_save_catalog_image_writes_to_class_dir(tmp_path: Path):
    with patch("rfconnectorai.data.scrape.requests.get") as mock_get:
        fake_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_get.return_value = MagicMock(
            status_code=200,
            content=fake_bytes,
            headers={"Content-Type": "image/png"},
        )
        img = CatalogImage(
            url="https://example.com/foo.png",
            class_name="SMA-M",
            filename="foo.png",
        )
        out_path = save_catalog_image(img, root=tmp_path)

    assert out_path.exists()
    assert out_path.parent.name == "SMA-M"
    assert out_path.read_bytes() == fake_bytes


def test_save_catalog_image_rejects_non_image_content_type(tmp_path: Path):
    with patch("rfconnectorai.data.scrape.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            content=b"<!doctype html>",
            headers={"Content-Type": "text/html"},
        )
        img = CatalogImage(
            url="https://example.com/foo.html",
            class_name="SMA-M",
            filename="foo.html",
        )
        with pytest.raises(ValueError):
            save_catalog_image(img, root=tmp_path)


def test_save_catalog_image_404_raises(tmp_path: Path):
    with patch("rfconnectorai.data.scrape.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=404, content=b"", headers={})
        img = CatalogImage(url="https://x/y", class_name="SMA-M", filename="y.png")
        with pytest.raises(RuntimeError):
            save_catalog_image(img, root=tmp_path)
