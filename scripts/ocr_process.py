#!/usr/bin/env python3
"""
OCR script for processing 金のフレーズ vocabulary book photos.
Uses EasyOCR for Japanese + English text recognition.

Usage:
    python ocr_process.py image1.jpg image2.jpg
    python ocr_process.py --dir ./photos/
    python ocr_process.py --dir ./photos/ --raw   # 生のOCR結果も出力

Output:
    data/words.json
"""

import json
import re
import sys
import argparse
from pathlib import Path


def setup_reader():
    import easyocr
    print("EasyOCRリーダーを初期化中... (初回はモデルダウンロードのため数分かかります)")
    return easyocr.Reader(['ja', 'en'], gpu=False)


def process_image(reader, image_path):
    print(f"処理中: {image_path}")
    results = reader.readtext(str(image_path), detail=1, paragraph=False)
    return results


def parse_results(results):
    lines = []
    for (bbox, text, confidence) in results:
        text = text.strip()
        if text and confidence > 0.3:
            lines.append({
                'text': text,
                'confidence': round(confidence, 3),
                'y': bbox[0][1],
            })
    lines.sort(key=lambda x: x['y'])
    return lines


def is_english(text):
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    english_count = sum(1 for c in alpha_chars if c.isascii())
    return english_count / len(alpha_chars) > 0.7


def build_word_pairs(lines):
    """
    テキスト行のリストから英単語・日本語訳のペアを構築する。
    金のフレーズのレイアウト: 番号 + 英語 → 次の行に日本語訳
    """
    words = []
    i = 0
    while i < len(lines):
        text = lines[i]['text']

        # 番号付きエントリーを検出 (例: "0001 abandon" or "001 account for")
        numbered = re.match(r'^(\d{1,4})\s+(.+)', text)
        if numbered:
            entry_id = int(numbered.group(1))
            english = numbered.group(2).strip()
            japanese = ''
            if i + 1 < len(lines) and not is_english(lines[i + 1]['text']):
                i += 1
                japanese = lines[i]['text']
            words.append({
                'id': entry_id,
                'english': english,
                'japanese': japanese,
                'confidence': lines[i]['confidence'],
            })
            i += 1
            continue

        # 番号なしの英語行 → 次行が日本語なら対にする
        if is_english(text) and len(text) > 1:
            english = text
            japanese = ''
            if i + 1 < len(lines) and not is_english(lines[i + 1]['text']):
                i += 1
                japanese = lines[i]['text']
            words.append({
                'id': len(words) + 1,
                'english': english,
                'japanese': japanese,
                'confidence': lines[i]['confidence'],
            })

        i += 1
    return words


def main():
    parser = argparse.ArgumentParser(description='金のフレーズ OCRスクリプト')
    parser.add_argument('images', nargs='*', help='処理する画像ファイル')
    parser.add_argument('--dir', help='画像ファイルのディレクトリ')
    parser.add_argument('--raw', action='store_true', help='生のOCR結果も出力する (デバッグ用)')
    parser.add_argument('--out', default='data/words.json', help='出力JSONファイルパス (デフォルト: data/words.json)')
    args = parser.parse_args()

    image_paths = []
    if args.dir:
        d = Path(args.dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(sorted(d.glob(ext)))
    for p in args.images:
        image_paths.append(Path(p))

    if not image_paths:
        print("エラー: 画像ファイルを指定してください")
        print("使用例:")
        print("  python ocr_process.py image1.jpg image2.jpg")
        print("  python ocr_process.py --dir ./photos/")
        sys.exit(1)

    reader = setup_reader()

    all_words = []
    all_raw = []

    for image_path in image_paths:
        if not image_path.exists():
            print(f"警告: ファイルが見つかりません: {image_path}")
            continue

        results = process_image(reader, image_path)
        lines = parse_results(results)
        words = build_word_pairs(lines)
        all_words.extend(words)
        print(f"  → {len(words)} 単語を検出")

        if args.raw:
            all_raw.append({'file': str(image_path), 'lines': lines})

    # ID を通し番号に振り直す
    for idx, word in enumerate(all_words):
        word['id'] = idx + 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_words = [
        {
            'id': w['id'],
            'english': w['english'],
            'japanese': w['japanese'],
            'partOfSpeech': '',
            'example': '',
        }
        for w in all_words
    ]

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_words, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {len(output_words)} 単語を {out_path} に保存しました")

    if args.raw:
        raw_path = out_path.with_suffix('.raw.json')
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(all_raw, f, ensure_ascii=False, indent=2)
        print(f"生データ: {raw_path} に保存しました")

    print("\n--- プレビュー (最初の10件) ---")
    for w in output_words[:10]:
        print(f"  [{w['id']:>4}] {w['english']:<30} → {w['japanese']}")


if __name__ == '__main__':
    main()
