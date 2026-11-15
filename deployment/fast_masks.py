"""2026-11-13 第一階段後處理加速: retina_masks 的原圖解析度 mask 只在每顆牙的框內計算。

ultralytics 的 ops.process_mask_native 會把 N 個 mask 全部雙線性放大到原圖(N × H × W 浮點)再用框裁掉,
筆電 CPU 上佔第一階段約 2/3 的時間(stage1_split.py)。框外本來就會被清成 0,所以只算框內的像素,
插值公式與 F.interpolate(bilinear, align_corners=False)相同,結果應該逐像素相同(只差浮點捨入)。

用法: fast_masks.install()(把 ultralytics.utils.ops.process_mask_native 換掉);install(check=True) 會兩種都算並統計不同的像素數。
驗證: python -m deployment.fast_masks → 開發集 303 張逐像素比較 + 牙位一致率,輸出 compare_results/deploy_eval/fast_masks/check.json
"""
import math

import torch
from ultralytics.utils import ops

ORIGINAL = ops.process_mask_native
STATS = {"calls": 0, "masks": 0, "pixels": 0, "diff_pixels": 0, "masks_with_diff": 0}


def process_mask_native_fast(protos, masks_in, bboxes, shape):
    c, mh, mw = protos.shape
    h, w = shape
    n = masks_in.shape[0]
    out = torch.zeros((n, h, w), dtype=torch.uint8, device=masks_in.device)
    if n == 0:
        return out
    low = (masks_in @ protos.float().view(c, -1)).view(n, mh, mw)
    # 與 ops.scale_masks 相同的 letterbox 邊界
    gain = min(mh / h, mw / w)
    pad_w, pad_h = (mw - round(w * gain)) / 2, (mh - round(h * gain)) / 2
    top, left = round(pad_h - 0.1), round(pad_w - 0.1)
    low = low[:, top:top + round(h * gain), left:left + round(w * gain)]
    sh, sw = low.shape[1:]

    def axis(a, b, src, dst):
        # F.interpolate(bilinear, align_corners=False)的來源座標與權重
        # 2026-11-15: 座標建在 mask 所在的裝置上(--device cuda 時 protos 在 GPU;CPU 上結果不變)
        s = ((torch.arange(a, b, dtype=torch.float32, device=low.device) + 0.5) * (src / dst) - 0.5).clamp_(min=0)
        i0 = s.floor().long().clamp_(max=src - 1)
        return i0, (i0 + 1).clamp_(max=src - 1), s - i0

    for i, (x1, y1, x2, y2) in enumerate(bboxes.tolist()):
        xa, xb = max(0, math.ceil(x1)), min(w, math.ceil(x2))  # crop_mask: x1 <= 欄 < x2
        ya, yb = max(0, math.ceil(y1)), min(h, math.ceil(y2))
        if xb <= xa or yb <= ya:
            continue
        y0, y1i, wy = axis(ya, yb, sh, h)
        x0, x1i, wx = axis(xa, xb, sw, w)
        m = low[i]
        r0, r1 = m[y0], m[y1i]
        val = (r0[:, x0] * (1 - wx) + r0[:, x1i] * wx) * (1 - wy)[:, None] + (r1[:, x0] * (1 - wx) + r1[:, x1i] * wx) * wy[:, None]
        out[i, ya:yb, xa:xb] = val.gt(0)
    return out


def checked(protos, masks_in, bboxes, shape):
    a = ORIGINAL(protos, masks_in, bboxes.clone(), shape)
    b = process_mask_native_fast(protos, masks_in, bboxes, shape)
    d = (a != b).flatten(1).sum(1)
    STATS["calls"] += 1
    STATS["masks"] += len(d)
    STATS["pixels"] += int(a.sum())
    STATS["diff_pixels"] += int(d.sum())
    STATS["masks_with_diff"] += int((d > 0).sum())
    return b


def install(check=False):
    ops.process_mask_native = checked if check else process_mask_native_fast


def uninstall():
    ops.process_mask_native = ORIGINAL


if __name__ == "__main__":
    import json

    import stage1.stage1_int8 as S
    out = S.ROOT / "compare_results" / "deploy_eval" / "fast_masks"
    out.mkdir(parents=True, exist_ok=True)
    ref = S.build_ref(S.dev_set())
    s1 = S.Stage1()
    install(check=True)
    r, _ = S.evaluate(s1, ref)
    res = {"dev_agreement": r, "pixel_check": STATS}
    (out / "check.json").write_text(json.dumps(res, indent=1))
    print(res)
