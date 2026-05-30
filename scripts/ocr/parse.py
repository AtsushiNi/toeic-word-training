import re
from collections import Counter

from .constants import ENTRIES_PER_PAGE, ENTRY_RE
from .utils import is_english


def detect_annotation_col_x(lines):
    """
    右ページの注釈カラム開始x座標を動的に推定する。

    ● / ■ マーカーで始まる行はほぼ必ず注釈カラムの先頭に現れる。
    これらの最小x座標を注釈カラム境界とする。
    マーカー行が見つからない場合は None を返し、フィルタリングを行わない。
    """
    MARKERS = ('●', '■', '◆', '▶')
    marker_xs = [l['x'] for l in lines if l['text'].strip().startswith(MARKERS)]
    if not marker_xs:
        return None
    # 最小x座標をそのまま使うとノイズに弱いため、下位20%ile を採用する
    marker_xs.sort()
    idx = max(0, int(len(marker_xs) * 0.20) - 1)
    return marker_xs[idx]


def is_inline_annotation(text):
    """
    右ページ左カラム内の補足注釈行かどうかを判定する。
    ※ 始まりの行は japanese の意味ではなく補足説明なので除外する。
    """
    return text.strip().startswith('※')


def is_headword(line):
    """
    見出し語かどうかを判定する。
    金のフレーズの見出し語は全て小文字・1〜3単語・高さ28px以上という特徴を持つ。
    """
    text = line['text'].strip()
    # 日本語・CJK文字を含む行は除外
    if any('　' <= c <= '鿿' or '＀' <= c <= '￯' for c in text):
        return False
    # 全て小文字でなければならない（発音記号等は大文字混在）
    if text != text.lower():
        return False
    # 1〜3単語のみ
    words = text.split()
    if not (1 <= len(words) <= 3):
        return False
    # 英字・スペース・ハイフンのみ
    if not re.match(r'^[a-z][a-z\s\-]*$', text):
        return False
    # 最小高さ: 小さな発音・注釈テキストを除外
    if line['h'] < 28:
        return False
    return True


def parse_left(lines, row_bands=None):
    """
    左ページから {id, exampleJa, exampleEnRaw} を ENTRIES_PER_PAGE 件固定で返す。
    row_bands が指定された場合は枠線検出結果のy範囲を各スロットに使用する。
    row_bands なしの場合はOCR検出IDからy範囲を推定し、欠損スロットは None で埋める。
    """
    xs = [l['x'] for l in lines]
    LEFT_PAGE_MAX_X = max(xs) * 0.75 if xs else float('inf')
    left_lines = [l for l in lines if l['x'] < LEFT_PAGE_MAX_X]

    # エントリー番号マーカーを収集してbase_idを特定
    markers = []
    for l in left_lines:
        m = ENTRY_RE.match(l['text'])
        if m:
            markers.append({'id': int(m.group(1)), 'y': l['y']})
    if not markers:
        return [None] * ENTRIES_PER_PAGE

    # 各マーカーIDが示すbase_idを多数決で決定し、外れ値マーカーを除去する。
    # 例: スプレッドの本来IDが021-030の場合、"021"の誤読"02"(id=2, base=1)より
    #     "022"-"030"(base=21)が9票多いためbase_id=21と正しく判定できる。
    implied = [((mk['id'] - 1) // ENTRIES_PER_PAGE) * ENTRIES_PER_PAGE + 1
               for mk in markers]
    base_id = Counter(implied).most_common(1)[0][0]
    # ベースIDの範囲外のマーカー（ページ番号・誤読など）を除去
    markers = [mk for mk in markers if base_id <= mk['id'] < base_id + ENTRIES_PER_PAGE]

    marker_by_id = {mk['id']: mk for mk in markers}

    slot_entries = []
    for i in range(ENTRIES_PER_PAGE):
        entry_id = base_id + i

        if row_bands is not None:
            # 枠線検出結果のy範囲を優先使用
            y_start, y_end = row_bands[i]
        elif entry_id in marker_by_id:
            # OCR検出IDマーカーからy範囲を推定
            marker  = marker_by_id[entry_id]
            y_start = marker['y'] - 5
            y_end   = float('inf')
            for j in range(i + 1, ENTRIES_PER_PAGE):
                next_id = base_id + j
                if next_id in marker_by_id:
                    y_end = marker_by_id[next_id]['y'] - 5
                    break
        else:
            # IDが検出されずrow_bandsもない場合はスキップ
            slot_entries.append(None)
            continue

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


def parse_right(lines, row_bands=None):
    """
    右ページから {english, japanese} を ENTRIES_PER_PAGE 件固定で返す。
    row_bands が指定された場合、各バンド内で見出し語と日本語意味を直接検索する。
    row_bands なしの場合はy座標のギャップ分析で欠損スロットを検出する。

    注釈カラム（● ■ 印の詳細解説）が存在する場合は動的に境界を検出して除外する。
    """
    if not lines:
        return [None] * ENTRIES_PER_PAGE

    # 注釈カラム境界を動的に検出し、境界より右のブロックを除外する
    annot_x = detect_annotation_col_x(lines)
    if annot_x is not None:
        lines = [l for l in lines if l['x'] < annot_x]

    if row_bands is not None:
        # 枠線検出バンドを使用して各スロットを処理
        entries = []
        for y_start, y_end in row_bands:
            band = [l for l in lines if y_start <= l['y'] <= y_end]
            band.sort(key=lambda l: l['y'])

            # Visionの言語補正で大文字化されることがあるため、小文字化してから判定する
            def _find_headword(lines):
                for l in lines:
                    if is_headword(l):
                        return l
                    lc = dict(l, text=l['text'].lower())
                    if is_headword(lc):
                        return lc
                return None
            hw      = _find_headword(band)
            english = hw['text'].strip() if hw else ''

            # 品詞マーカー付きの日本語意味を優先して探す（※注釈行は除外）
            ja_band  = [l for l in band if not is_english(l['text']) and not is_inline_annotation(l['text'])]
            japanese = ''
            for l in ja_band:
                t = l['text'].strip()
                if t and t[0] in '名動形副前接間代助' and len(t) > 2:
                    japanese = t[1:].lstrip()
                    break
            if not japanese:
                for l in sorted(ja_band, key=lambda x: x['y']):
                    t = l['text'].strip()
                    if not is_english(t) and len(t) > 3:
                        japanese = t
                        break

            entries.append(dict(english=english, japanese=japanese) if english else None)
        return entries

    # row_bands なし: 既存のy座標ギャップ分析ロジックを使用
    headwords = [l for l in lines if is_headword(l)]
    headwords.sort(key=lambda l: l['y'])

    if not headwords:
        return [None] * ENTRIES_PER_PAGE

    # y座標のギャップ分析で欠損スロットにNoneを挿入する
    placed = list(headwords)
    if len(placed) >= 2:
        gaps        = [placed[i + 1]['y'] - placed[i]['y'] for i in range(len(placed) - 1)]
        median_gap  = sorted(gaps)[len(gaps) // 2]

        i = len(placed) - 1
        while i > 0:
            gap     = placed[i]['y'] - placed[i - 1]['y']
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
        y_end   = hw['y'] + 300
        for next_hw in placed[slot_idx + 1:]:
            if next_hw is not None:
                y_end = next_hw['y'] - 5
                break

        # ※注釈行を除いた日本語バンドで意味を検索する
        band = [l for l in lines
                if y_start <= l['y'] <= y_end
                and not is_english(l['text'])
                and not is_inline_annotation(l['text'])]
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


def fill_blank(raw, word):
    """
    例文中の空欄プレースホルダーを実際の単語で置換する。
    OCRのばらつき（'a-------'/'a-'/'a.'/'a'）に対応する。
    """
    if not raw or not word:
        return raw or ''
    # 先頭文字 + 任意のダッシュ/ドットのパターンを単語境界で置換
    pat    = re.compile(r'\b' + re.escape(word[0]) + r'[-\.]*(?=\s|$)', re.IGNORECASE)
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
        base_id     = ((min_left_id - 1) // ENTRIES_PER_PAGE) * ENTRIES_PER_PAGE + 1
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
