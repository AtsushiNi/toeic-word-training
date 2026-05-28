# TOEIC Word Training

「金のフレーズ」の写真を OCR で読み取り、Expo + React Native の単語学習アプリで学べるようにするプロジェクトです。

```
[金のフレーズ写真] → OCRスクリプト → words.json → モバイルアプリ（フラッシュカード学習）
```

## プロジェクト構成

| コンポーネント | 概要 | 状態 |
|---|---|---|
| `scripts/ocr_process.py` | 写真から単語データを抽出する OCR スクリプト | 実装済み |
| Expo / React Native アプリ | フラッシュカード学習・単語一覧画面 | 開発中 |

---

## OCR スクリプト

### 動作環境

- **OS**: macOS（Apple Vision Framework を使用するため必須）
- **Python**: 3.9 以上
- **依存ライブラリ**: `pyobjc-framework-Vision`, `Pillow`

### インストール

```bash
pip install pyobjc-framework-Vision Pillow
```

### 入力画像の準備

**ディレクトリ構成（新形式）**

左ページと右ページをそれぞれ別ディレクトリに入れます。ファイル名順にペアを組むため、ファイル名の並び順を左右で一致させてください。

```
data/
├── 左ページ/
│   ├── 001.jpg
│   └── ...
└── 右ページ/
    ├── 001.jpg
    └── ...
```

**フラット形式（旧形式）**

```
data/
├── left1.jpg
├── right1.jpg
└── ...
```

撮影時は明るく均一な光で撮影し、ページ全体が垂直に収まるようにしてください。

### 実行

```bash
# 基本実行（data/words.json に出力）
python scripts/ocr_process.py --dir ./data/

# 生 OCR データも保存（data/words.raw.json）
python scripts/ocr_process.py --dir ./data/ --raw

# 出力先を指定
python scripts/ocr_process.py --dir ./data/ --out ./data/my_words.json
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--dir` | `data` | 入力画像のディレクトリ |
| `--out` | `data/words.json` | 出力 JSON のパス |
| `--raw` | なし | 生 OCR データを `*.raw.json` に追加保存 |

### 出力フォーマット（words.json）

```json
[
  {
    "id": 1,
    "english": "anyway",
    "japanese": "とにかく",
    "partOfSpeech": "副",
    "level": 600,
    "exampleJa": "とにかくやってみよう。",
    "exampleEn": "Let's try anyway."
  }
]
```

| フィールド | 型 | 説明 |
|---|---|---|
| `id` | number | 通し番号（1始まり） |
| `english` | string | 見出し語 |
| `japanese` | string | 日本語訳 |
| `partOfSpeech` | string | 品詞（`名`動`形`副`前`接`間`代`助`） |
| `level` | number | TOEIC スコア目標レベル |
| `exampleJa` | string | 例文（日本語） |
| `exampleEn` | string | 例文（英語） |

---

## モバイルアプリ（開発中）

Expo + React Native で実装予定。`words.json` を読み込み、以下の機能を提供します。

- 単語一覧画面
- フラッシュカード学習画面
