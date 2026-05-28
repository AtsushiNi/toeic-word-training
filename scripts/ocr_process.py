#!/usr/bin/env python3
"""
OCR script for 金のフレーズ using Apple Vision Framework (macOS only).
Requires: pip install pyobjc-framework-Vision Pillow numpy

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

from PIL import Image

from ocr.vision import ocr_file
from ocr.preprocess import normalize_brightness, detect_row_bands
from ocr.parse import parse_left, parse_right, merge


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
    d         = Path(directory)
    left_dir  = d / '左ページ'
    right_dir = d / '右ページ'

    if left_dir.is_dir() and right_dir.is_dir():
        exts        = {'jpg', 'jpeg', 'png'}
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
            num   = re.search(r'\d+', left.stem)
            if not num:
                continue
            right = left.with_name(f'right{num.group()}.{ext}')
            if right.exists():
                pairs.append((left, right))
    return pairs


def process_spread(left_path, right_path):
    print(f"処理中: {left_path.name} + {right_path.name}")

    # 元画像を読み込み、枠線検出と明るさ正規化を実施してからOCRに渡す
    left_img_orig  = Image.open(left_path)
    right_img_orig = Image.open(right_path)

    print("  前処理中 (枠線検出・明るさ正規化)...")
    left_bands,  left_reliable  = detect_row_bands(left_img_orig)
    right_bands, right_reliable = detect_row_bands(right_img_orig)

    # 信頼性が低い場合（ボーダー間隔のばらつきが大きい）はギャップ分析にフォールバック
    if not right_reliable:
        print("  右ページ: 枠線ばらつき大 → ギャップ分析を使用")
        right_bands = None

    left_img_norm  = normalize_brightness(left_img_orig)
    right_img_norm = normalize_brightness(right_img_orig)

    print("  左ページ OCR...")
    lines_left = ocr_file(left_img_norm)
    print(f"    {len(lines_left)} テキストブロック検出")

    print("  右ページ OCR...")
    lines_right = ocr_file(right_img_norm)
    print(f"    {len(lines_right)} テキストブロック検出")

    left_entries  = parse_left(lines_left,  row_bands=left_bands)
    right_entries = parse_right(lines_right, row_bands=right_bands)
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
