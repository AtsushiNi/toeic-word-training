"""
ocr/parse.py のユニットテスト。
Apple Vision不要・画像不要で実行可能。
"""
import sys
import os
import unittest

# scripts/ をパスに追加してパッケージを解決する
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ocr.parse import is_headword, parse_left, parse_right, fill_blank, merge
from ocr.constants import ENTRIES_PER_PAGE
from ocr.utils import is_english


def _line(text, x=50, y=100, w=100, h=30, conf=0.9):
    """テスト用OCRラインを生成するヘルパー。"""
    return dict(text=text, confidence=conf, x=x, y=y, w=w, h=h)


class TestIsEnglish(unittest.TestCase):
    def test_英語のみ(self):
        self.assertTrue(is_english("anyway"))

    def test_日本語のみ(self):
        self.assertFalse(is_english("とにかく"))

    def test_空文字(self):
        self.assertFalse(is_english(""))

    def test_数字のみ(self):
        self.assertFalse(is_english("123"))

    def test_混在_英語多数(self):
        self.assertTrue(is_english("Let's try"))

    def test_混在_日本語多数(self):
        self.assertFalse(is_english("動 試みる"))


class TestIsHeadword(unittest.TestCase):
    def test_正常な見出し語(self):
        self.assertTrue(is_headword(_line("anyway", h=35)))

    def test_複数単語の見出し語(self):
        self.assertTrue(is_headword(_line("take part", h=35)))

    def test_大文字混在は除外(self):
        self.assertFalse(is_headword(_line("Anyway", h=35)))

    def test_4単語以上は除外(self):
        self.assertFalse(is_headword(_line("take part in this", h=35)))

    def test_高さ不足は除外(self):
        self.assertFalse(is_headword(_line("anyway", h=20)))

    def test_日本語含むは除外(self):
        self.assertFalse(is_headword(_line("test テスト", h=35)))

    def test_数字含むは除外(self):
        self.assertFalse(is_headword(_line("test1", h=35)))

    def test_ハイフン区切りは許可(self):
        self.assertTrue(is_headword(_line("up-to-date", h=35)))


class TestFillBlank(unittest.TestCase):
    def test_ダッシュプレースホルダーを置換(self):
        # 末尾のドットも[-\.]でマッチするため、ピリオドごと置換される
        self.assertEqual(fill_blank("Let's try a-------.", "anyway"), "Let's try anyway")

    def test_短いダッシュ(self):
        self.assertEqual(fill_blank("I a- anyway.", "am"), "I am anyway.")

    def test_ドット区切り(self):
        # 末尾のドットも[-\.]でマッチするため、ピリオドごと置換される
        self.assertEqual(fill_blank("It's a.", "anyway"), "It's anyway")

    def test_単語なしは空文字(self):
        self.assertEqual(fill_blank("", "anyway"), "")

    def test_rawなしは空文字(self):
        self.assertEqual(fill_blank(None, "anyway"), "")

    def test_単語なしはrawをそのまま返す(self):
        self.assertEqual(fill_blank("Let's try.", ""), "Let's try.")

    def test_一致しない先頭文字は置換しない(self):
        result = fill_blank("Let's try z-------.", "anyway")
        self.assertNotIn("anyway", result)


class TestParseLeft(unittest.TestCase):
    def _make_left_lines(self, base_id=1):
        """基本的な左ページOCRデータを生成する。"""
        lines = []
        for i in range(ENTRIES_PER_PAGE):
            entry_id = base_id + i
            y_base = i * 100
            lines.append(_line(str(entry_id), x=10, y=y_base))
            lines.append(_line(f"例文日本語{i}", x=50, y=y_base + 20))
            lines.append(_line(f"English example {i}.", x=50, y=y_base + 50))
        return lines

    def test_基本的なパース(self):
        lines = self._make_left_lines(base_id=1)
        result = parse_left(lines)
        self.assertEqual(len(result), ENTRIES_PER_PAGE)
        non_null = [e for e in result if e is not None]
        self.assertGreater(len(non_null), 0)

    def test_IDが正しく振られる(self):
        lines = self._make_left_lines(base_id=11)
        result = parse_left(lines)
        non_null = [e for e in result if e is not None]
        ids = [e['id'] for e in non_null]
        self.assertEqual(min(ids), 11)

    def test_マーカーなしはNone列を返す(self):
        lines = [_line("英語テキスト", x=50, y=100)]
        result = parse_left(lines)
        self.assertEqual(result, [None] * ENTRIES_PER_PAGE)

    def test_row_bandsありでパース(self):
        lines = self._make_left_lines(base_id=1)
        bands = [(i * 100, (i + 1) * 100) for i in range(ENTRIES_PER_PAGE)]
        result = parse_left(lines, row_bands=bands)
        self.assertEqual(len(result), ENTRIES_PER_PAGE)


class TestParseRight(unittest.TestCase):
    def _make_right_lines(self):
        """基本的な右ページOCRデータを生成する。"""
        lines = []
        words = ["anyway", "achieve", "affect", "agree", "allow",
                 "apply", "approach", "argue", "avoid", "believe"]
        for i, word in enumerate(words):
            y_base = i * 120
            lines.append(_line(word, x=50, y=y_base, h=35))
            lines.append(_line(f"副 {word}の意味", x=50, y=y_base + 40))
        return lines

    def test_基本的なパース(self):
        lines = self._make_right_lines()
        result = parse_right(lines)
        self.assertEqual(len(result), ENTRIES_PER_PAGE)

    def test_見出し語が取得できる(self):
        lines = self._make_right_lines()
        result = parse_right(lines)
        non_null = [e for e in result if e is not None]
        self.assertGreater(len(non_null), 0)
        self.assertEqual(non_null[0]['english'], 'anyway')

    def test_空行の場合はNone列を返す(self):
        result = parse_right([])
        self.assertEqual(result, [None] * ENTRIES_PER_PAGE)

    def test_row_bandsありでパース(self):
        lines = self._make_right_lines()
        bands = [(i * 120, (i + 1) * 120) for i in range(ENTRIES_PER_PAGE)]
        result = parse_right(lines, row_bands=bands)
        self.assertEqual(len(result), ENTRIES_PER_PAGE)

    def test_大文字化された見出し語をフォールバック検出(self):
        # Vision OCRが大文字化した場合でも検出できるか
        lines = [_line("ANYWAY", x=50, y=0, h=35)]
        bands = [(0, 200)]
        result = parse_right(lines, row_bands=bands)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['english'], 'anyway')


class TestMerge(unittest.TestCase):
    def _make_left(self, base_id=1):
        # exampleEnRaw の先頭文字を単語の先頭文字（w）に合わせてfill_blankが機能するようにする
        return [
            dict(id=base_id + i, exampleJa=f"日本語例文{i}", exampleEnRaw=f"English w---.")
            for i in range(ENTRIES_PER_PAGE)
        ]

    def _make_right(self, words=None):
        if words is None:
            words = [f"word{i}" for i in range(ENTRIES_PER_PAGE)]
        return [
            dict(english=w, japanese=f"{w}の意味")
            for w in words
        ]

    def test_基本的なマージ(self):
        result = merge(self._make_left(1), self._make_right())
        self.assertEqual(len(result), ENTRIES_PER_PAGE)
        self.assertEqual(result[0]['id'], 1)
        self.assertEqual(result[0]['english'], 'word0')
        self.assertIn('word0', result[0]['exampleEn'])

    def test_IDが正しく振られる(self):
        result = merge(self._make_left(11), self._make_right())
        self.assertEqual(result[0]['id'], 11)
        self.assertEqual(result[9]['id'], 20)

    def test_左がNoneの場合は位置からID推定(self):
        left = [None] * ENTRIES_PER_PAGE
        right = self._make_right()
        result = merge(left, right)
        self.assertEqual(result[0]['id'], 1)

    def test_右がNoneの場合は空文字(self):
        left = self._make_left(1)
        right = [None] * ENTRIES_PER_PAGE
        result = merge(left, right)
        self.assertEqual(result[0]['english'], '')
        self.assertEqual(result[0]['japanese'], '')

    def test_結果にexampleEnが含まれる(self):
        # exampleEnRaw は "English w---." → 先頭文字 w で始まる単語 "word" で置換される
        result = merge(self._make_left(1), self._make_right(['word'] * ENTRIES_PER_PAGE))
        self.assertIn('word', result[0]['exampleEn'])


if __name__ == '__main__':
    unittest.main()
