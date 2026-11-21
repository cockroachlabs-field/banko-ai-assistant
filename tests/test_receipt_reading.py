"""Reading a receipt's text: a page whose first reading holds no amount is read once more.

A faded thermal receipt, scanned to PDF, was refused: "We couldn't find the
total on that receipt". The total was on the page. Left to lay the page out
by itself, tesseract dropped that line and several others; told that the
page is one block of text, which a receipt is, it read "$46.29".

Sharpening, thresholding or thickening the image did worse than nothing: each
gave a total that was wrong and looked right (46.40, 46.49, 46.79). So the
image is left alone and only the way the page is laid out changes."""

import io
import json
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from banko_ai.agents.tools import document_tools as D

BY_ITSELF = "ARCO GASOLINE\n9200 AVIATION BLVD\nAuth #: 093291\nTHANK YOU\n"          # the line with the total is gone
AS_ONE_BLOCK = "ARCO GASOLINE\n9200 AVIATION BLVD\nVISA $46.29\nAuth #: 093291\nTHANK YOU\n"


class _Tesseract:
    """Stands in for pytesseract: what it reads depends on the page mode it is asked for."""

    def __init__(self, by_itself: str, as_one_block: str):
        self.readings, self.asked = {"": by_itself, "--psm 6": as_one_block}, []

    def image_to_string(self, image, config: str = "") -> str:
        self.asked.append(config)
        return self.readings[config]


def test_a_page_whose_first_reading_holds_no_amount_is_read_again_as_one_block(monkeypatch):
    tesseract = _Tesseract(BY_ITSELF, AS_ONE_BLOCK)
    monkeypatch.setattr(D, "pytesseract", tesseract)
    assert D.read_text(object()) == AS_ONE_BLOCK
    assert tesseract.asked == ["", "--psm 6"]


def test_a_page_that_reads_well_is_read_once(monkeypatch):
    tesseract = _Tesseract("BLUE BOTTLE COFFEE\nLatte 5.50\nTOTAL 9,75\n", "never asked for")
    monkeypatch.setattr(D, "pytesseract", tesseract)
    assert D.read_text(object()) == "BLUE BOTTLE COFFEE\nLatte 5.50\nTOTAL 9,75\n"      # a comma for the point still counts
    assert tesseract.asked == [""]


def test_a_second_reading_without_an_amount_changes_nothing(monkeypatch):
    """Not a receipt at all, or one nobody could read: the first reading stands."""
    tesseract = _Tesseract("a page of words\nphone 1-800-322-2726\n", "other words\n")
    monkeypatch.setattr(D, "pytesseract", tesseract)
    assert D.read_text(object()) == "a page of words\nphone 1-800-322-2726\n"
    assert tesseract.asked == ["", "--psm 6"]


def _tools() -> dict:
    return {tool.name: tool for tool in D.create_document_tools("postgresql://unused", None)}


def _blank_png(path) -> str:
    Image.new("RGB", (300, 400), "white").save(path, "PNG")
    return str(path)


def test_an_image_upload_is_read_that_way(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "pytesseract", _Tesseract(BY_ITSELF, AS_ONE_BLOCK))
    out = json.loads(_tools()["extract_text_from_image"].func(_blank_png(tmp_path / "r.png")))
    assert out["success"] and "$46.29" in out["text"]


@pytest.mark.skipif(not shutil.which("pdftoppm"), reason="poppler is not installed")
def test_a_scanned_pdf_is_read_that_way(monkeypatch, tmp_path):
    """A scan has no text layer: every page is a picture."""
    scan = tmp_path / "scan.pdf"
    Image.new("RGB", (600, 900), "white").save(scan, "PDF")
    monkeypatch.setattr(D, "pytesseract", _Tesseract(BY_ITSELF, AS_ONE_BLOCK))
    out = json.loads(_tools()["extract_text_from_pdf"].func(str(scan)))
    assert out["success"] and "$46.29" in out["text"]


@pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract is not installed")
def test_a_clean_receipt_reads_as_it_always_did():
    """With the real tesseract: one reading, every line."""
    lines = ["BLUE BOTTLE COFFEE", "300 Webster St, Oakland CA", "Latte              5.50", "TOTAL              9.75"]
    try:
        font = ImageFont.truetype("Arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default(size=40)
    image = Image.new("RGB", (900, 120 + 60 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((60, 60 + 60 * i), line, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    text = D.read_text(Image.open(io.BytesIO(buffer.getvalue())))
    assert "BLUE BOTTLE COFFEE" in text and "9.75" in text and "5.50" in text
