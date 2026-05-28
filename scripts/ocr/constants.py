import re

# 各ページの固定エントリー数（金のフレーズ仕様）
ENTRIES_PER_PAGE = 10

# エントリーID番号にマッチする正規表現（先頭ゼロ許容・1〜3桁）
ENTRY_RE = re.compile(r'^0*(\d{1,3})$')
