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
import os
import re
import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
from Foundation import NSURL
import Vision
from PIL import Image, ImageFilter

from ocr.constants import ENTRIES_PER_PAGE
from ocr.parse import parse_left, parse_right, merge


# ── Apple Vision OCR ─────────────────────────────────────────────────────────

def ocr_file(image_source):
    """
    Apple Vision で精度最高モードのOCRを実行する。
    image_source にはファイルパスまたは PIL.Image を渡せる。
    返り値: {text, confidence, x, y, w, h} のリスト（y昇順、top-left基準）
    """
    tmp_path = None
    try:
        if isinstance(image_source, (str, Path)):
            path = str(Path(image_source).resolve())
            img_w, img_h = Image.open(image_source).size
        else:
            # PIL Image は一時ファイルに保存してから Vision に渡す
            img_w, img_h = image_source.size
            fd, tmp_path = tempfile.mkstemp(suffix='.png')
            os.close(fd)
            image_source.save(tmp_path, format='PNG')
            path = tmp_path

        url = NSURL.fileURLWithPath_(path)

        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLanguages_(["ja-JP", "en-US"])
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            print(f"  Vision エラー: {err}")
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

            # Vision: 正規化座標・左下原点 → ピクセル・左上原点に変換
            bb   = obs.boundingBox()
            x    = float(bb.origin.x) * img_w
            y_bl = float(bb.origin.y) * img_h
            w    = float(bb.size.width)  * img_w
            h    = float(bb.size.height) * img_h
            y    = img_h - y_bl - h

            lines.append(dict(text=text, confidence=round(conf, 3),
                              x=round(x, 1), y=round(y, 1),
                              w=round(w, 1), h=round(h, 1)))

        lines.sort(key=lambda l: l['y'])
        return lines

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── 画像前処理 ────────────────────────────────────────────────────────────────

def normalize_brightness(img):
    """
    大きなガウシアンブラーで背景輝度を推定し、画像全体の明るさムラを補正する。
    撮影時のライト位置によるグラデーション状の照明ムラに効果的。
    """
    gray = img.convert('L')
    # 短辺の1/8をぼかし半径として照明ムラのスケールを推定する
    radius = min(img.width, img.height) // 8
    bg = gray.filter(ImageFilter.GaussianBlur(radius=radius))

    arr    = np.array(img, dtype=np.float64)
    bg_arr = np.array(bg,  dtype=np.float64)

    # 背景輝度を目標値（192）に統一する補正係数を算出
    target = 192.0
    scale  = target / np.where(bg_arr > 1.0, bg_arr, 1.0)
    # 過補正（暗すぎる・明るすぎる補正）を防ぐためクリップ
    scale  = np.clip(scale, 0.5, 3.0)

    if arr.ndim == 3:
        corrected = np.clip(arr * scale[:, :, np.newaxis], 0, 255).astype(np.uint8)
    else:
        corrected = np.clip(arr * scale, 0, 255).astype(np.uint8)

    return Image.fromarray(corrected)


def detect_row_bands(img, n_rows=None):
    """
    各行の推定中心位置の近傍で最暗行を探してボーダーとし、
    n_rows 行分のバンド [(y_start, y_end), ...] を返す。
    行が等間隔に並ぶという前提を利用するため、薄い枠線でも安定して検出できる。
    """
    if n_rows is None:
        n_rows = ENTRIES_PER_PAGE

    gray   = np.array(img.convert('L'), dtype=np.float32)
    img_h, img_w = gray.shape

    # 左右10%を除いた中央部の各行平均輝度を計算（端の影・綴じ目ノイズを排除）
    margin         = img_w // 10
    row_brightness = gray[:, margin:img_w - margin].mean(axis=1)

    # 均等間隔のボーダー推定位置の近傍（±行高さの30%）で最暗行を探す
    row_height    = img_h / n_rows
    search_radius = int(row_height * 0.3)

    borders = []
    for k in range(1, n_rows):
        expected_y = int(k * row_height)
        y_lo = max(0,     expected_y - search_radius)
        y_hi = min(img_h, expected_y + search_radius)
        local_min_idx = int(np.argmin(row_brightness[y_lo:y_hi]))
        borders.append(y_lo + local_min_idx)

    # ボーダーからバンドを生成（先頭は画像上端、末尾は画像下端まで）
    all_y = [0] + borders + [img_h]
    bands = [(all_y[k], all_y[k + 1]) for k in range(n_rows)]

    # ボーダー間隔の変動係数（CV）で信頼性を判定する
    # CV = 標準偏差 / 平均。画像によって枠線の見え方が異なる場合に高くなる
    gaps = [borders[i + 1] - borders[i] for i in range(len(borders) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    cv = (sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean_gap
    reliable = cv < 0.15  # 変動15%未満を信頼できる検出とみなす

    return bands, reliable


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
