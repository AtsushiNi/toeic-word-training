import os
import tempfile
from pathlib import Path

from Foundation import NSURL
import Vision
from PIL import Image


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
