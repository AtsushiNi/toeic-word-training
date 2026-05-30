import json
import re
import sys
import argparse
from pathlib import Path

from PIL import Image

from .vision import ocr_file
from .preprocess import normalize_brightness, detect_row_bands, detect_page_region, deskew
from .parse import parse_left, parse_right, merge


def find_spreads(directory):
    """
    ディレクトリ内の左右ページペアを検索して返す。

    新形式: <directory>/左ページ/ + <directory>/右ページ/ ディレクトリ
      ファイル名昇順でソートし、i番目の左ページとi番目の右ページをペアにする。

    旧形式: left1.jpg + right1.jpg, left2.jpg + right2.jpg, …

    ソート済みの (left_path, right_path) タプルのリストを返す。
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


def preprocess_spread(left_path, right_path, save_dir=None):
    """
    左右ページ画像に前処理（ページ検出・傾き補正・明るさ正規化）を適用して返す。
    save_dir が指定された場合は前処理後の画像を PNG で保存する。
    保存先: <save_dir>/左ページ/<元ファイル名>.png, <save_dir>/右ページ/<元ファイル名>.png
    """
    left_img  = Image.open(left_path)
    right_img = Image.open(right_path)

    # ページ領域を検出して透視変換で正規化する
    left_img,  left_page_found  = detect_page_region(left_img)
    right_img, right_page_found = detect_page_region(right_img)

    # ページ検出できなかった場合はデスキューにフォールバック
    if not left_page_found:
        left_img  = deskew(left_img)
    if not right_page_found:
        right_img = deskew(right_img)

    left_img  = normalize_brightness(left_img)
    right_img = normalize_brightness(right_img)

    if save_dir is not None:
        left_out  = Path(save_dir) / '左ページ' / (left_path.stem  + '.png')
        right_out = Path(save_dir) / '右ページ' / (right_path.stem + '.png')
        left_out.parent.mkdir(parents=True, exist_ok=True)
        right_out.parent.mkdir(parents=True, exist_ok=True)
        left_img.save(left_out,   format='PNG')
        right_img.save(right_out, format='PNG')
        print(f"  前処理後画像を保存: {left_out}, {right_out}")

    return left_img, right_img


def process_spread(left_path, right_path, use_preprocess=True, preprocess_save_dir=None):
    """1ページスプレッド（左右ペア）を前処理・OCR・パースしてエントリーリストを返す。"""
    print(f"処理中: {left_path.name} + {right_path.name}")

    if use_preprocess:
        print("  前処理中 (ページ検出・傾き補正・明るさ正規化)...")
        left_img, right_img = preprocess_spread(left_path, right_path, save_dir=preprocess_save_dir)
    else:
        print("  前処理スキップ...")
        left_img  = Image.open(left_path)
        right_img = Image.open(right_path)

    # 前処理後の画像で行バンドを検出する
    left_bands,  left_reliable  = detect_row_bands(left_img)
    right_bands, right_reliable = detect_row_bands(right_img)

    # 信頼性が低い場合（ボーダー間隔のばらつきが大きい）はフォールバックを使用
    # 左ページ: None にするとIDマーカーのy座標ベースの範囲推定を使用する
    if not left_reliable:
        print("  左ページ: 枠線ばらつき大 → IDマーカーベースの解析を使用")
        left_bands = None
    # 右ページ: None にするとy座標ギャップ分析を使用する
    if not right_reliable:
        print("  右ページ: 枠線ばらつき大 → ギャップ分析を使用")
        right_bands = None

    print("  左ページ OCR...")
    lines_left = ocr_file(left_img)
    print(f"    {len(lines_left)} テキストブロック検出")

    print("  右ページ OCR...")
    lines_right = ocr_file(right_img)
    print(f"    {len(lines_right)} テキストブロック検出")

    left_entries  = parse_left(lines_left,  row_bands=left_bands)
    right_entries = parse_right(lines_right, row_bands=right_bands)
    entries       = merge(left_entries, right_entries)

    print(f"  → {len(entries)} エントリー")
    return entries, dict(left=lines_left, right=lines_right)


def main():
    """メインエントリポイント。CLIの引数を解析してOCR処理を実行する。"""
    parser = argparse.ArgumentParser(description='金のフレーズ OCR (Apple Vision)')
    parser.add_argument('--dir', default='data', help='leftN/rightN 画像のディレクトリ')
    parser.add_argument('--raw', action='store_true', help='生OCRデータも保存')
    parser.add_argument('--out', default='data/words.json')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='ページ検出・傾き補正・明るさ正規化をスキップして生画像のままOCRを実行する')
    parser.add_argument('--save-preprocessed', action='store_true',
                        help='前処理後の画像を <dir>/preprocessed/ に PNG で保存する')
    parser.add_argument('--stage', choices=['preprocess'],
                        help='実行するステージを指定する。preprocess: 前処理のみ実行して画像を保存')
    args = parser.parse_args()

    spreads = find_spreads(args.dir)
    if not spreads:
        print(
            f"エラー: {args.dir}/ に画像ペアが見つかりません\n"
            f"  新形式: {args.dir}/左ページ/ + {args.dir}/右ページ/ ディレクトリ\n"
            f"  旧形式: {args.dir}/left1.jpg + {args.dir}/right1.jpg"
        )
        sys.exit(1)

    # --stage preprocess: 前処理のみ実行して終了
    if args.stage == 'preprocess':
        preprocess_out = Path(args.dir) / 'preprocessed'
        print(f"前処理ステージ: {len(spreads)} ペアを処理して {preprocess_out} に保存します")
        for left_path, right_path in spreads:
            print(f"処理中: {left_path.name} + {right_path.name}")
            preprocess_spread(left_path, right_path, save_dir=preprocess_out)
        print(f"\n完了: {len(spreads)} ペアの前処理後画像を {preprocess_out} に保存しました")
        return

    # --save-preprocessed: 前処理後画像の保存先を設定する
    preprocess_save_dir = Path(args.dir) / 'preprocessed' if args.save_preprocessed else None

    all_entries = []
    all_raw     = []

    for left_path, right_path in spreads:
        entries, raw = process_spread(
            left_path, right_path,
            use_preprocess=not args.no_preprocess,
            preprocess_save_dir=preprocess_save_dir,
        )
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

    if preprocess_save_dir:
        print(f"前処理後画像: {preprocess_save_dir}")

    print("\n--- プレビュー (最初の10件) ---")
    for e in all_entries[:10]:
        print(f"  [{e['id']:>3}] {e['english']:<20} {e['japanese']}")
        if e['exampleJa']:
            print(f"        JA: {e['exampleJa']}")
        if e['exampleEn']:
            print(f"        EN: {e['exampleEn']}")
