# SPDX-License-Identifier: Apache-2.0

"""Exercise actual encoded inputs and reject TIFF before Grana decodes pixels."""
import hashlib
from io import BytesIO
import json
import struct

import numpy as np
from PIL import Image, TiffImagePlugin
import pytest

from grana.inference.cli import load_request
from grana.inference.client import build_binary
from grana.inference.predictor import InferenceRequest, _observations
from grana.inference.presentation import render_prediction
from scripts import annotation_skill_api as evidence
from test_annotation_skill_api import inputs  # noqa: F401


TIFF_VARIANTS = [("<", False), (">", False), ("<", True), (">", True)]
# Pillow 12.0 recognizes the big-endian BigTIFF prefix but cannot open that variant.
# File-input rejection still exercises all four variants before invoking Pillow.
PIL_TIFF_VARIANTS = TIFF_VARIANTS[:3]
RGB = (10, 20, 30)


def _tiff_bytes(order, big):
    """A small valid uncompressed RGB TIFF, without an optional encoder dependency."""
    def pack(spec, *values):
        return struct.pack(order + spec, *values)

    marker = b"II" if order == "<" else b"MM"
    header = marker + (pack("HHHQ", 43, 8, 0, 16) if big else pack("HI", 42, 8))
    value_size = 8 if big else 4
    ifd_end = len(header) + (8 + 10 * 20 + 8 if big else 2 + 10 * 12 + 4)
    pixels_offset = ifd_end if big else ifd_end + 6
    bits = pack("HHH", 8, 8, 8) if big else pack("I", ifd_end)
    entries = [(256, 4, 1, pack("I", 4)), (257, 4, 1, pack("I", 3)),
               (258, 3, 3, bits), (259, 3, 1, pack("H", 1)),
               (262, 3, 1, pack("H", 2)), (273, 4, 1, pack("I", pixels_offset)),
               (277, 3, 1, pack("H", 3)), (278, 4, 1, pack("I", 3)),
               (279, 4, 1, pack("I", 36)), (284, 3, 1, pack("H", 1))]
    directory = pack("Q" if big else "H", len(entries))
    for tag, kind, count, value in entries:
        directory += pack("HHQ" if big else "HHI", tag, kind, count) + value.ljust(value_size, b"\0")
    directory += pack("Q" if big else "I", 0)
    return header + directory + (b"" if big else pack("HHH", 8, 8, 8)) + bytes(RGB) * 12


def _request(frame):
    return InferenceRequest(images={"top": [frame] * 8}, compressed_task_prompt="Move the block.")


def _consume(boundary, request, tmp_path):
    if boundary == "predictor":
        return _observations(request, None)
    if boundary == "client":
        return build_binary(request, roles=["A"])
    result = {"roles": ["A"], "poi_slot": 5, "progress": {"A": {"values": [.5] * 8}}}
    return render_prediction(result, request, tmp_path / "plot.png")


@pytest.mark.parametrize("order,big", PIL_TIFF_VARIANTS)
def test_tiff_fixtures_are_valid_rgb_images(order, big):
    with Image.open(BytesIO(_tiff_bytes(order, big))) as image:
        assert image.format == "TIFF" and image.mode == "RGB" and image.size == (4, 3)
        assert image.tobytes() == bytes(RGB) * 12


@pytest.mark.parametrize("order,big", TIFF_VARIANTS)
@pytest.mark.parametrize("extension", [".tiff", ".png", ".jpg", ".webp", ".bin"])
def test_tiff_file_inputs_are_rejected_before_pillow_open(tmp_path, monkeypatch, order, big, extension):
    frame = tmp_path / ("frame" + extension)
    frame.write_bytes(_tiff_bytes(order, big))
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": [frame.name] * 8}}))
    monkeypatch.setattr(Image, "open", lambda *a, **kw: pytest.fail("TIFF must not reach Image.open"))
    with pytest.raises(ValueError, match="TIFF image inputs"):
        load_request(path)


@pytest.mark.parametrize("prefix", [b"II*\0", b"MM\0*", b"II+\0", b"MM\0+", b"II\0*", b"MM*\0"])
def test_truncated_and_reversed_tiff_headers_are_rejected_before_pillow_open(tmp_path, monkeypatch, prefix):
    (tmp_path / "frame.png").write_bytes(prefix)
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": ["frame.png"] * 8}}))
    monkeypatch.setattr(Image, "open", lambda *a, **kw: pytest.fail("TIFF header must not reach Image.open"))
    with pytest.raises(ValueError, match="TIFF image inputs"):
        load_request(path)


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "BMP", "PPM"])
@pytest.mark.parametrize("name", ["frame.bin", "frame.tiff"])
def test_file_inputs_preserve_other_rgb_formats_independent_of_extension(tmp_path, format, name):
    frame = tmp_path / name
    Image.new("RGB", (4, 3), RGB).save(frame, format=format)
    with Image.open(frame) as image:
        expected = np.array(image)
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": [frame.name] * 8}}))
    request = load_request(path)
    for image in request.images["top"]:
        np.testing.assert_array_equal(image, expected)


@pytest.mark.parametrize("order,big", PIL_TIFF_VARIANTS)
@pytest.mark.parametrize("boundary", ["predictor", "client", "renderer"])
def test_lazy_pil_tiff_is_rejected_before_pixel_load(tmp_path, monkeypatch, order, big, boundary):
    with Image.open(BytesIO(_tiff_bytes(order, big))) as image:
        monkeypatch.setattr(image, "load", lambda *a, **kw: pytest.fail("TIFF pixels must not be loaded"))
        with pytest.raises(ValueError, match="TIFF image inputs"):
            _consume(boundary, _request(image), tmp_path)


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
@pytest.mark.parametrize("boundary", ["predictor", "client", "renderer"])
def test_lazy_pil_supported_formats_remain_accepted(tmp_path, format, boundary):
    encoded = BytesIO()
    Image.new("RGB", (4, 3), RGB).save(encoded, format=format)
    encoded.seek(0)
    with Image.open(encoded) as image:
        assert _consume(boundary, _request(image), tmp_path)


@pytest.mark.parametrize("order,big", TIFF_VARIANTS)
def test_annotation_rejects_renamed_tiff_before_tiff_plugin_opens(inputs, monkeypatch, order, big):
    args, _, manifest = inputs
    row = manifest["images"][0]
    raw = _tiff_bytes(order, big)
    (args["evidence_path"].parent / row["image_path"]).write_bytes(raw)
    row["sha256"] = hashlib.sha256(raw).hexdigest()
    args["evidence_path"].write_text(json.dumps(manifest))
    monkeypatch.setattr(TiffImagePlugin.TiffImageFile, "_open",
                        lambda *a, **kw: pytest.fail("Annotation must not invoke the TIFF plugin"))
    with pytest.raises(ValueError, match="cannot be decoded"):
        evidence.prepare_request(args["bundle_path"], args["evidence_path"], args["policy_dir"])


@pytest.mark.parametrize("format,mime", [("PNG", "png"), ("JPEG", "jpeg"), ("WEBP", "webp")])
def test_annotation_accepts_all_existing_formats(inputs, format, mime):
    args, _, manifest = inputs
    for row in manifest["images"]:
        path = args["evidence_path"].parent / row["image_path"]
        Image.new("RGB", (8, 4), RGB).save(path, format=format)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    args["evidence_path"].write_text(json.dumps(manifest))
    _, content, images = evidence.prepare_request(args["bundle_path"], args["evidence_path"], args["policy_dir"])
    assert len(images) == len(manifest["images"])
    assert all(row["image_url"].startswith(f"data:image/{mime};base64,")
               for row in content if row["type"] == "input_image")
