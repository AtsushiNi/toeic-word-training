def is_english(text):
    """テキストが英語（ASCII英字70%超）かどうかを判定する。"""
    alpha = [c for c in text if c.isalpha()]
    return bool(alpha) and sum(1 for c in alpha if c.isascii()) / len(alpha) > 0.7
