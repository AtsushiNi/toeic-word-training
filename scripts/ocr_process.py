#!/usr/bin/env python3
"""
OCR script for 金のフレーズ using Apple Vision Framework (macOS only).
Requires: pip install pyobjc-framework-Vision

Input naming convention:
  left1.jpg  + right1.jpg   → spread 1
  left2.jpg  + right2.jpg   → spread 2
  ...

Left page layout:
  [entry#]  Japanese example sentence
            English sentence with blank (e.g. "Let's try a-------.")

Right page layout:
  [large English word]
  [pronunciation]
  [品詞 Japanese meaning]
  [detailed notes ...]

Usage:
  python ocr_process.py --dir ./data/
  python ocr_process.py --dir ./data/ --raw
"""

import json
import re
import sys
import argparse
from pathlib import Path

from Foundation import NSURL
import Vision
from PIL import Image


# ── Apple Vision OCR ─────────────────────────────────────────────────────────

def ocr_file(image_path):
    """
    Run Apple Vision accurate OCR on image_path.
    Returns list of {text, confidence, x, y, w, h} sorted top-to-bottom.
    Coordinates are pixels with top-left origin.
    """
    path = str(Path(image_path).resolve())
    url  = NSURL.fileURLWithPath_(path)

    img_w, img_h = Image.open(image_path).size

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLanguages_(["ja-JP", "en-US"])
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        print(f"  Vision error: {err}")
        return []

    lines = []
    for obs in (req.results() or []):
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        top  = cands[0]
        text = str(top.string()).strip()
        conf = float(top.confidence())
        if not text or conf < 0.1:
            continue

        # Vision: normalized coords, bottom-left origin → convert to pixel top-left
        bb   = obs.boundingBox()
        x    = float(bb.origin.x) * img_w
        y_bl = float(bb.origin.y) * img_h
        w    = float(bb.size.width)  * img_w
        h    = float(bb.size.height) * img_h
        y    = img_h - y_bl - h      # flip to top-left origin

        lines.append(dict(text=text, confidence=round(conf, 3),
                          x=round(x, 1), y=round(y, 1),
                          w=round(w, 1), h=round(h, 1)))

    lines.sort(key=lambda l: l['y'])
    return lines


# ── helpers ──────────────────────────────────────────────────────────────────

def is_english(text):
    alpha = [c for c in text if c.isalpha()]
    return bool(alpha) and sum(1 for c in alpha if c.isascii()) / len(alpha) > 0.7

ENTRY_RE = re.compile(r'^0*(\d{1,3})$')
LEVEL_RE = re.compile(r'(?:Score\s*)?(\d{3,4})\s*[Ll]evel', re.IGNORECASE)
POS_RE   = re.compile(r'^[名動形副前接間代助]\s')


# ── left page ────────────────────────────────────────────────────────────────

def parse_left(lines):
    """
    Extract {id, exampleJa, exampleEnRaw} per entry from the left page.

    The left-page photo often shows the right-page edge (high x values).
    We restrict to the left 75% of the image to avoid that bleed-through.

    Each entry's y-range starts from the entry-number line and ends just before
    the next entry-number line.  We include lines at the same y as the entry
    number itself (the Japanese sentence is often on the same horizontal band).
    English fragments are sorted by x (left→right) for natural reading order.
    """
    xs = [l['x'] for l in lines]
    LEFT_PAGE_MAX_X = max(xs) * 0.75 if xs else float('inf')
    left_lines = [l for l in lines if l['x'] < LEFT_PAGE_MAX_X]

    # Collect all entry-number markers {id, y}
    markers = []
    for l in left_lines:
        m = ENTRY_RE.match(l['text'])
        if m:
            markers.append({'id': int(m.group(1)), 'y': l['y']})
    if not markers:
        return []

    entries = []
    for idx, marker in enumerate(markers):
        y_start = marker['y'] - 5           # include same-line text
        y_end   = markers[idx + 1]['y'] - 5 if idx + 1 < len(markers) else float('inf')

        band = [l for l in left_lines
                if y_start <= l['y'] < y_end
                and not ENTRY_RE.match(l['text'])]

        ex_ja    = ''
        en_frags = []   # list of (x, text) for left→right sorting
        for l in sorted(band, key=lambda l: l['y']):
            t = l['text'].strip()
            if not t:
                continue
            if not is_english(t) and not ex_ja:
                ex_ja = t
            elif is_english(t):
                en_frags.append((l['x'], t))

        # Sort EN fragments left→right so reading order is preserved
        en_frags.sort(key=lambda p: p[0])
        ex_en = ' '.join(t for _, t in en_frags).strip()

        entries.append(dict(id=marker['id'], exampleJa=ex_ja, exampleEnRaw=ex_en))

    return entries


# ── right page ───────────────────────────────────────────────────────────────

def is_headword(line):
    """
    Return True if this line looks like a vocabulary headword.
    Headwords in 金のフレーズ are:
      - All lowercase (displayed without capitalisation)
      - Short (1-3 words)
      - Purely alphabetic with optional hyphens/spaces
      - Tall enough to distinguish from small pronunciation text
    """
    text = line['text'].strip()
    # No Japanese or CJK characters
    if any('　' <= c <= '鿿' or '＀' <= c <= '￯' for c in text):
        return False
    # Must be all lowercase (pronunciations and other noise are mixed-case)
    if text != text.lower():
        return False
    # 1-3 words only
    words = text.split()
    if not (1 <= len(words) <= 3):
        return False
    # Only letters, spaces, hyphens
    if not re.match(r'^[a-z][a-z\s\-]*$', text):
        return False
    # Minimum height: filters out small pronunciation/annotation text
    if line['h'] < 28:
        return False
    return True


def parse_right(lines):
    """
    Extract {english, japanese, partOfSpeech, level} per entry from the right page.
    Headwords are short purely-alphabetic English lines (see is_headword()).
    """
    if not lines:
        return []

    # Score level
    current_level = None
    for l in lines:
        m = LEVEL_RE.search(l['text'])
        if m:
            current_level = int(m.group(1))
            break

    headwords = [l for l in lines if is_headword(l)]
    headwords.sort(key=lambda l: l['y'])

    entries = []
    for idx, hw in enumerate(headwords):
        y_start = hw['y'] - 5
        y_end   = headwords[idx + 1]['y'] - 5 if idx + 1 < len(headwords) else hw['y'] + 300

        # Collect non-English lines in this band for Japanese meaning
        band = [l for l in lines
                if y_start <= l['y'] <= y_end and not is_english(l['text'])]
        band.sort(key=lambda l: l['y'])

        japanese = ''
        pos      = ''
        # Priority 1: POS line with actual meaning content after the POS char
        for l in band:
            t = l['text'].strip()
            if t and t[0] in '名動形副前接間代助' and len(t) > 2:
                pos      = t[0]
                japanese = t[1:].lstrip()
                break
        # Priority 2: first non-English line with meaningful content (>3 chars)
        if not japanese:
            for l in sorted(band, key=lambda x: x['y']):
                t = l['text'].strip()
                if not is_english(t) and len(t) > 3:
                    japanese = t
                    break

        entries.append(dict(english=hw['text'].strip(), japanese=japanese,
                            partOfSpeech=pos, level=current_level))

    return entries


# ── merge ────────────────────────────────────────────────────────────────────

def fill_blank(raw, word):
    """Replace blank placeholder with the actual word.
    Handles OCR variations: 'a-------', 'a-', 'a.' or just 'a' at word boundary.
    """
    if not raw or not word:
        return raw or ''
    # Match first letter followed by optional dashes/dots at a word boundary
    pat = re.compile(r'\b' + re.escape(word[0]) + r'[-\.]*(?=\s|$)', re.IGNORECASE)
    result = pat.sub(word, raw, count=1)
    return result


def merge(left_entries, right_entries):
    left_by_id = {e['id']: e for e in left_entries}
    left_ids   = sorted(left_by_id)

    merged = []
    for idx, right in enumerate(right_entries):
        if idx < len(left_ids):
            lid  = left_ids[idx]
            left = left_by_id[lid]
        else:
            lid  = idx + 1
            left = dict(id=lid, exampleJa='', exampleEnRaw='')

        merged.append(dict(
            id           = lid,
            english      = right['english'],
            japanese     = right['japanese'],
            partOfSpeech = right['partOfSpeech'],
            level        = right['level'],
            exampleJa    = left.get('exampleJa', ''),
            exampleEn    = fill_blank(left.get('exampleEnRaw', ''), right['english']),
        ))
    return merged


# ── main ─────────────────────────────────────────────────────────────────────

def find_spreads(directory):
    """
    Find left/right page pairs in directory.
    Expects: left1.jpg + right1.jpg, left2.jpg + right2.jpg, …
    Returns sorted list of (left_path, right_path) tuples.
    """
    d = Path(directory)
    pairs = []
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
        for left in sorted(d.glob(f'left*.{ext}')):
            num    = re.search(r'\d+', left.stem)
            if not num:
                continue
            right = left.with_name(f'right{num.group()}.{ext}')
            if right.exists():
                pairs.append((left, right))
    return pairs


def process_spread(left_path, right_path):
    print(f"処理中: {left_path.name} + {right_path.name}")

    print("  左ページ OCR...")
    lines_left  = ocr_file(left_path)
    print(f"    {len(lines_left)} テキストブロック検出")

    print("  右ページ OCR...")
    lines_right = ocr_file(right_path)
    print(f"    {len(lines_right)} テキストブロック検出")

    left_entries  = parse_left(lines_left)
    right_entries = parse_right(lines_right)
    entries       = merge(left_entries, right_entries)

    print(f"  → {len(entries)} エントリー")
    return entries, dict(left=lines_left, right=lines_right)


def main():
    parser = argparse.ArgumentParser(description='金のフレーズ OCR (Apple Vision)')
    parser.add_argument('--dir', default='data', help='leftN/rightN 画像のディレクトリ')
    parser.add_argument('--raw', action='store_true', help='生OCRデータも保存')
    parser.add_argument('--out', default='data/words.json')
    args = parser.parse_args()

    spreads = find_spreads(args.dir)
    if not spreads:
        print(f"エラー: {args.dir}/ に left1.jpg + right1.jpg のペアが見つかりません")
        sys.exit(1)

    all_entries = []
    all_raw     = []

    for left_path, right_path in spreads:
        entries, raw = process_spread(left_path, right_path)
        all_entries.extend(entries)
        if args.raw:
            all_raw.append(dict(left_path=str(left_path), right_path=str(right_path), **raw))

    # 通し番号に振り直し
    for idx, e in enumerate(all_entries):
        e['id'] = idx + 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    print(f"\n完了: {len(all_entries)} エントリーを {out_path} に保存")

    if args.raw:
        raw_path = out_path.with_suffix('.raw.json')
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(all_raw, f, ensure_ascii=False, indent=2)
        print(f"生データ: {raw_path}")

    print("\n--- プレビュー (最初の10件) ---")
    for e in all_entries[:10]:
        print(f"  [{e['id']:>3}] {e['english']:<20} {e['partOfSpeech']} {e['japanese']}")
        if e['exampleJa']:
            print(f"        JA: {e['exampleJa']}")
        if e['exampleEn']:
            print(f"        EN: {e['exampleEn']}")


if __name__ == '__main__':
    main()
