import numpy as np
from PIL import Image, ImageFilter

from .constants import ENTRIES_PER_PAGE


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
