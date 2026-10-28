import numpy as np

# FDI 編號 v2(2026-10-25,實驗中)。中線的部分(每側不超過 8 顆)已併入 fdi_cropped.assign_fdi_indices;
#   空隙判斷仍在實驗中,fdi_v2_check.py --gap <比例> 與目前版本比較。
# 舊版的兩個弱點(stage1_failures 目視確認):
#   1. 中線: 只用「面積總和最小的連續 4 顆」當門牙,缺牙或植體時會找錯,曾把 11 顆牙編到同一側(410、411)。
#   2. 從中線依序編號,不考慮缺牙的空隙: 缺 34 時 35 會被編成 34;牙齒沒偵測到時也一樣。
# v2:
#   中線: 與舊版相同(在中間 80% 的牙齒中,找面積總和最小的連續 4 顆,中線在第 2、3 顆之間),
#         但只考慮兩側都不超過 8 顆的位置(舊版會編出 39、410 這種不存在的編號)。
#         CENTER_WEIGHT > 0 時再加上「離影像中央的距離」;第一版用 4.0,目視發現會把正確的中線拉偏一顆,改為預設 0。
#   編號: GAP_RATIO 不是 None 時,從中線往遠心數,相鄰兩顆牙(或中線到第一顆)的空隙 > GAP_RATIO × 兩顆平均寬度,
#         視為中間少了 round(空隙 / 平均寬度) 顆(缺牙或沒偵測到),編號跳過;超過 8 的牙齒不編號。
# 寬度用多邊形外框(bbox)的 x 範圍;回傳格式與舊版相同(每顆牙加上 quadrant、fdi_id)。
CENTER_WEIGHT = 0.0
GAP_RATIO = None  # 空隙判斷改動太多,要等 FDI / 牙位 GT 才能判斷好壞,預設關閉
MAX_PER_SIDE = 8


def width(t):
    return t["bbox"][2] - t["bbox"][0]


def find_midline(teeth, image_width, center_weight=CENTER_WEIGHT):
    n = len(teeth)
    if n < 4:
        return (teeth[0]["cx"] + teeth[-1]["cx"]) / 2.0
    med_area = float(np.median([t["area"] for t in teeth]))
    start, end = int(n * 0.1), int(n * 0.9) - 3
    if end < start:
        start, end = 0, n - 3
    best, best_cost = None, float("inf")
    for i in range(0, n - 3):
        left, right = i + 2, n - i - 2
        if left > MAX_PER_SIDE or right > MAX_PER_SIDE:
            continue
        mid = (teeth[i + 1]["cx"] + teeth[i + 2]["cx"]) / 2.0
        cost = sum(teeth[j]["area"] for j in range(i, i + 4)) / (4 * med_area) + center_weight * abs(mid - image_width / 2) / image_width
        # 舊版只在中間 80% 找;超出範圍的位置只有在範圍內都不合法時才用
        if not start <= i <= end:
            cost += 100
        if cost < best_cost:
            best, best_cost = mid, cost
    return best if best is not None else image_width / 2


def number_side(side_teeth, mesial_edge, gap_ratio):
    # side_teeth 依離中線由近到遠排序;mesial_edge(t) 回傳 (近中線邊界, 遠中線邊界) 到中線的距離
    numbers, prev_far, prev_w, n = [], 0.0, None, 0
    for t in side_teeth:
        near, far = mesial_edge(t)
        w = width(t)
        ref = w if prev_w is None else (w + prev_w) / 2
        gap = near - prev_far
        missing = int(round(gap / ref)) if gap_ratio is not None and gap > gap_ratio * ref else 0
        n += 1 + missing
        numbers.append(n)
        prev_far, prev_w = far, w
    return numbers


def assign_fdi_v2(sorted_teeth, image_width, center_weight=CENTER_WEIGHT, gap_ratio=GAP_RATIO):
    if not sorted_teeth:
        return []
    teeth = sorted(sorted_teeth, key=lambda t: t["cx"])
    mid_x = find_midline(teeth, image_width, center_weight)
    for t in teeth:
        t.pop("fdi_id", None)
        t["quadrant"] = 4 if t["cx"] <= mid_x else 3
    # 象限 4 在影像左側: 離中線的距離 = mid_x − x
    q4 = sorted([t for t in teeth if t["quadrant"] == 4], key=lambda t: -t["cx"])
    for t, n in zip(q4, number_side(q4, lambda t: (mid_x - t["bbox"][2], mid_x - t["bbox"][0]), gap_ratio)):
        if n <= MAX_PER_SIDE:
            t["fdi_id"] = f"4{n}"
    q3 = sorted([t for t in teeth if t["quadrant"] == 3], key=lambda t: t["cx"])
    for t, n in zip(q3, number_side(q3, lambda t: (t["bbox"][0] - mid_x, t["bbox"][2] - mid_x), gap_ratio)):
        if n <= MAX_PER_SIDE:
            t["fdi_id"] = f"3{n}"
    return teeth
