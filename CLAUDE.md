# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## タスクを進める時の注意
最初に最新のリモートブランチを確認すること

## コーディング規約

コメントは日本語で丁寧に記述すること。

## Overview

This project OCR-processes photos of the TOEIC vocabulary book 「金のフレーズ」 using Apple Vision Framework (macOS only) and outputs structured JSON for vocabulary training.

## Running the script

```bash
# Standard run (outputs data/words.json)
python scripts/ocr_process.py --dir ./data/

# Also save raw OCR output (data/words.raw.json)
python scripts/ocr_process.py --dir ./data/ --raw

# Custom output path
python scripts/ocr_process.py --dir ./data/ --out ./data/my_words.json

# Skip preprocessing (page detection / deskew / brightness normalization)
python scripts/ocr_process.py --dir ./data/ --no-preprocess

# Run preprocessing only and save images to data/preprocessed/
python scripts/ocr_process.py --dir ./data/ --stage preprocess

# Full run + save preprocessed images to data/preprocessed/
python scripts/ocr_process.py --dir ./data/ --save-preprocessed
```

## Dependencies

macOS-only. Install with:
```bash
pip install pyobjc-framework-Vision Pillow numpy opencv-python
```

## Input image layout

The script supports two input modes (auto-detected):

**Directory mode (新形式):** `data/左ページ/` + `data/右ページ/` — files sorted by filename, i-th left pairs with i-th right.

**Flat mode (旧形式):** `data/left1.jpg` + `data/right1.jpg`, `left2.jpg` + `right2.jpg`, etc.

## Output format

`data/words.json` — array of entries:
```json
{
  "id": 1,
  "english": "anyway",
  "japanese": "とにかく",
  "partOfSpeech": "副",
  "level": 600,
  "exampleJa": "とにかくやってみよう。",
  "exampleEn": "Let's try anyway."
}
```

`partOfSpeech` uses single Japanese characters: `名`(noun) `動`(verb) `形`(adj) `副`(adv) `前`(prep) `接`(conj) `間`(interj) `代`(pron) `助`(aux).

## Architecture

All logic is in `scripts/ocr_process.py`:

- `ocr_file()` — runs Apple Vision on an image, returns bounding-box-annotated text lines sorted top-to-bottom
- `parse_left()` — extracts entry IDs + Japanese/English example sentences from left pages; clips to left 75% of image to avoid right-page bleed-through
- `parse_right()` — extracts headwords (detected by `is_headword()`: lowercase, 1–3 words, height ≥ 28px), Japanese meanings, and TOEIC score level
- `merge()` — zips left and right entries by index (not by entry number) and calls `fill_blank()` to substitute the headword into the example sentence blank
- `find_spreads()` — detects which input mode is in use and returns sorted `(left_path, right_path)` pairs

Preprocessing pipeline (`scripts/ocr/preprocess.py`, applied before OCR by default):
- `detect_page_region()` — Canny edge detection + contour approximation to find the page quadrilateral; applies perspective transform to straighten and crop to the page area; returns `(image, detected: bool)`
- `deskew()` — fallback when page detection fails; estimates tilt angle from text-row bounding boxes and rotates to correct
- `normalize_brightness()` — estimates background illumination via large Gaussian blur and normalizes to a uniform brightness
- `detect_row_bands()` — locates the 10 horizontal row dividers and returns band coordinates used by the parser

`data/` is gitignored for raw photos (`data/raw/`) but `words.json` / `words.raw.json` are committed.
