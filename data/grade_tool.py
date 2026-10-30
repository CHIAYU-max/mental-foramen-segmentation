import argparse
import csv

import cv2

from data.visibility_grading import GRADES_CSV, IMAGES

# 頦孔可辨識性的鍵盤判定工具:一次顯示一張匿名影像,按鍵判定後自動跳到下一張,結果即時寫入 grades.csv
#   1 = 看得出來    0 = 看不出來    b = 回上一張    q = 結束(已判定的會保留,下次從未判定的開始)
SCALE = 2
KEYS = {ord("1"): "1", ord("0"): "0"}


def save(order, grades):
    with open(GRADES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "mf_visible"])
        for alias in order:
            writer.writerow([alias, grades.get(alias, "")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", action="store_true", help="從第一張開始,包含已判定的影像")
    args = parser.parse_args()

    if not GRADES_CSV.exists():
        raise SystemExit(f"找不到 {GRADES_CSV},請先執行 python -m data.visibility_grading prepare")
    with open(GRADES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    order = [r["alias"] for r in rows]
    grades = {r["alias"]: r["mf_visible"] for r in rows if r["mf_visible"].strip() in ("0", "1")}
    paths = {p.stem: p for p in IMAGES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}

    i = 0 if args.redo else next((k for k, a in enumerate(order) if a not in grades), len(order))
    window = "mf visibility  [1] visible  [0] not visible  [b] back  [q] quit"
    while 0 <= i < len(order):
        alias = order[i]
        image = cv2.imread(str(paths[alias]))
        image = cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        current = grades.get(alias, "-")
        cv2.putText(image, f"{alias}  {i + 1}/{len(order)}  done {len(grades)}  current: {current}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            i = max(0, i - 1)
            continue
        if key in KEYS:
            grades[alias] = KEYS[key]
            save(order, grades)
            i += 1
    cv2.destroyAllWindows()
    save(order, grades)
    print(f"已判定 {len(grades)} / {len(order)} 張,結果存在 {GRADES_CSV}")


if __name__ == "__main__":
    main()
