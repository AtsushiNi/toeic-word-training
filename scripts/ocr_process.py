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
ENTRIES_PER_PAGE = 10  # 各ページ固定行数（金のフレーズ仕様）


# ── left page ────────────────────────────────────────────────────────────────

def parse_left(lines):
    """
    左ページから {id, exampleJa, exampleEnRaw} を ENTRIES_PER_PAGE 件固定で返す。
    OCRで検出できなかったスロットは None で埋め、IDのずれを防ぐ。
    """
    xs = [l['x'] for l in lines]
    LEFT_PAGE_MAX_X = max(xs) * 0.75 if xs else float('inf')
    left_lines = [l for l in lines if l['x'] < LEFT_PAGE_MAX_X]

    # エントリー番号マーカーを収集
    markers = []
    for l in left_lines:
        m = ENTRY_RE.match(l['text'])
        if m:
            markers.append({'id': int(m.group(1)), 'y': l['y']})
    if not markers:
        return [None] * ENTRIES_PER_PAGE

    # 検出した最小IDからページの開始IDを特定（例: 1, 11, 21 ...）
    min_id = min(mk['id'] for mk in markers)
    base_id = ((min_id - 1) // ENTRIES_PER_PAGE) * ENTRIES_PER_PAGE + 1
    marker_by_id = {mk['id']: mk for mk in markers}

    slot_entries = []
    for i in range(ENTRIES_PER_PAGE):
        entry_id = base_id + i
        if entry_id not in marker_by_id:
            slot_entries.append(None)
            continue

        marker = marker_by_id[entry_id]

        # 次に存在するマーカーのy座標を終端とする
        y_end = float('inf')
        for j in range(i + 1, ENTRIES_PER_PAGE):
            next_id = base_id + j
            if next_id in marker_by_id:
                y_end = marker_by_id[next_id]['y'] - 5
                break

        y_start = marker['y'] - 5

        band = [l for l in left_lines
                if y_start <= l['y'] < y_end
                and not ENTRY_RE.match(l['text'])]

        ex_ja    = ''
        en_frags = []
        for l in sorted(band, key=lambda l: l['y']):
            t = l['text'].strip()
            if not t:
                continue
            if not is_english(t) and not ex_ja:
                ex_ja = t
            elif is_english(t):
                en_frags.append((l['x'], t))

        en_frags.sort(key=lambda p: p[0])
        ex_en = ' '.join(t for _, t in en_frags).strip()

        slot_entries.append(dict(id=entry_id, exampleJa=ex_ja, exampleEnRaw=ex_en))

    return slot_entries


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
    右ページから {english, japanese} を ENTRIES_PER_PAGE 件固定で返す。
    y座標のギャップから欠損行を検出し、欠損スロットは None で埋める。
    """
    if not lines:
        return [None] * ENTRIES_PER_PAGE

    headwords = [l for l in lines if is_headword(l)]
    headwords.sort(key=lambda l: l['y'])

    if not headwords:
        return [None] * ENTRIES_PER_PAGE

    # y座標のギャップ分析で欠損スロットにNoneを挿入する
    placed = list(headwords)
    if len(placed) >= 2:
        gaps = [placed[i + 1]['y'] - placed[i]['y'] for i in range(len(placed) - 1)]
        median_gap = sorted(gaps)[len(gaps) // 2]

        # 後ろから走査してギャップが大きい箇所にNoneを挿入
        i = len(placed) - 1
        while i > 0:
            gap = placed[i]['y'] - placed[i - 1]['y']
            missing = round(gap / median_gap) - 1
            for _ in range(missing):
                placed.insert(i, None)
            i -= 1

    # ENTRIES_PER_PAGE 件になるよう末尾をNoneで埋める
    while len(placed) < ENTRIES_PER_PAGE:
        placed.append(None)
    placed = placed[:ENTRIES_PER_PAGE]

    entries = []
    for slot_idx, hw in enumerate(placed):
        if hw is None:
            entries.append(None)
            continue

        y_start = hw['y'] - 5
        y_end = hw['y'] + 300
        for next_hw in placed[slot_idx + 1:]:
            if next_hw is not None:
                y_end = next_hw['y'] - 5
                break

        band = [l for l in lines
                if y_start <= l['y'] <= y_end and not is_english(l['text'])]
        band.sort(key=lambda l: l['y'])

        japanese = ''
        for l in band:
            t = l['text'].strip()
            if t and t[0] in '名動形副前接間代助' and len(t) > 2:
                japanese = t[1:].lstrip()
                break
        if not japanese:
            for l in sorted(band, key=lambda x: x['y']):
                t = l['text'].strip()
                if not is_english(t) and len(t) > 3:
                    japanese = t
                    break

        entries.append(dict(english=hw['text'].strip(), japanese=japanese))

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
    """
    左右ページの ENTRIES_PER_PAGE 件リストを位置で対応させてマージする。
    左ページのIDを優先し、欠損の場合は位置から推定する。
    """
    # base_idを左ページの非Noneエントリーから特定
    non_null_left = [e for e in left_entries if e is not None]
    if non_null_left:
        min_left_id = min(e['id'] for e in non_null_left)
        base_id = ((min_left_id - 1) // ENTRIES_PER_PAGE) * ENTRIES_PER_PAGE + 1
    else:
        base_id = 1

    merged = []
    for idx in range(ENTRIES_PER_PAGE):
        left  = left_entries[idx]  if idx < len(left_entries)  else None
        right = right_entries[idx] if idx < len(right_entries) else None

        entry_id  = left['id']           if left  is not None else base_id + idx
        english   = right['english']     if right is not None else ''
        japanese  = right['japanese']    if right is not None else ''
        ex_ja     = left['exampleJa']    if left  is not None else ''
        ex_en_raw = left['exampleEnRaw'] if left  is not None else ''

        merged.append(dict(
            id        = entry_id,
            english   = english,
            japanese  = japanese,
            exampleJa = ex_ja,
            exampleEn = fill_blank(ex_en_raw, english),
        ))
    return merged


# ── main ─────────────────────────────────────────────────────────────────────

def find_spreads(directory):
    """
    Find left/right page pairs in directory.

    Mode 1 – directory-based (新形式):
      <directory>/左ページ/*.jpg  +  <directory>/右ページ/*.jpg
      Files are sorted by filename ascending; i-th left pairs with i-th right.

    Mode 2 – flat naming (既存形式):
      left1.jpg + right1.jpg, left2.jpg + right2.jpg, …

    Returns sorted list of (left_path, right_path) tuples.
    """
    d = Path(directory)
    left_dir  = d / '左ページ'
    right_dir = d / '右ページ'

    if left_dir.is_dir() and right_dir.is_dir():
        exts = {'jpg', 'jpeg', 'png'}
        left_files  = sorted(
            f for f in left_dir.iterdir()
            if f.suffix.lower().lstrip('.') in exts
        )
        right_files = sorted(
            f for f in right_dir.iterdir()
            if f.suffix.lower().lstrip('.') in exts
        )
        if len(left_files) != len(right_files):
            print(
                f"エラー: 左ページ ({len(left_files)}枚) と"
                f" 右ページ ({len(right_files)}枚) の枚数が一致しません"
            )
            sys.exit(1)
        return list(zip(left_files, right_files))

    pairs = []
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
        for left in sorted(d.glob(f'left*.{ext}')):
            num = re.search(r'\d+', left.stem)
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
        print(
            f"エラー: {args.dir}/ に画像ペアが見つかりません\n"
            f"  新形式: {args.dir}/左ページ/ + {args.dir}/右ページ/ ディレクトリ\n"
            f"  旧形式: {args.dir}/left1.jpg + {args.dir}/right1.jpg"
        )
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
        print(f"  [{e['id']:>3}] {e['english']:<20} {e['japanese']}")
        if e['exampleJa']:
            print(f"        JA: {e['exampleJa']}")
        if e['exampleEn']:
            print(f"        EN: {e['exampleEn']}")


if __name__ == '__main__':
    main()
