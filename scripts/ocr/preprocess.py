import cv2
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


def detect_page_region(img):
    """
    Canny エッジ検出と輪郭近似でページ領域（最大四角形）を検出し、
    透視変換で正規化したクロップ画像を返す。
    四角形が見つかった場合は (変換後画像, True)、失敗時は (元画像, False) を返す。
    """
    arr  = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr.copy()
    h, w = gray.shape

    # ぼかしてノイズを除去してからエッジ検出
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    # エッジを膨張させて輪郭を繋げる
    dilated   = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, False

    # 面積の大きい順に上位5件を候補として四角形を探す
    page_pts = None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        # 画像面積の30%未満は対象外
        if cv2.contourArea(cnt) < 0.3 * h * w:
            break
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            page_pts = approx.reshape(4, 2).astype(np.float32)
            break

    if page_pts is None:
        return img, False

    # 4点を左上・右上・右下・左下の順に並べ替え
    s    = page_pts.sum(axis=1)
    diff = np.diff(page_pts, axis=1).ravel()
    tl   = page_pts[np.argmin(s)]
    br   = page_pts[np.argmax(s)]
    tr   = page_pts[np.argmin(diff)]
    bl   = page_pts[np.argmax(diff)]
    src  = np.array([tl, tr, br, bl], dtype=np.float32)

    # 出力サイズ: 上下辺・左右辺それぞれの長い方を採用
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)

    M      = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, M, (out_w, out_h))
    return Image.fromarray(warped), True


def deskew(img):
    """
    Otsu 二値化後に水平方向に膨張してテキスト行を塊として検出し、
    minAreaRect で傾き角を推定して回転補正する。
    ページ領域検出に失敗した場合のフォールバックとして使用する。
    傾きが 0.5 度未満の場合は元画像をそのまま返す。
    """
    arr  = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr.copy()

    # Otsu 二値化でテキスト領域を白、背景を黒に反転
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 水平方向に膨張してテキスト行を1つの塊にまとめる
    dil_w  = max(1, gray.shape[1] // 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, 1))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area    = gray.shape[0] * gray.shape[1] * 0.001

    angles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect  = cv2.minAreaRect(cnt)
        angle = rect[2]
        # minAreaRect は [-90, 0) を返す。水平に近い矩形の角度に統一
        if angle < -45:
            angle += 90
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return img

    skew = float(np.median(angles))
    if abs(skew) < 0.5:  # 0.5度未満は補正不要
        return img

    h, w   = gray.shape
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, skew, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)
