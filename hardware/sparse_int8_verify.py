import argparse
import csv
import time
import warnings

import torch

from deployment.int8_quant import read_split
from hardware.qat_fx import OUT, convert, save_masks, to_fixed
from hardware.qat_int4 import prepare

warnings.filterwarnings("ignore")
# 2:4 + INT8 的整數與定點驗證(2026-11-10,伺服器 CPU,使用者要求補完整流程;見 EXPERIMENTS.md「硬體: 結構化稀疏與 4-bit 的評估與規劃」)。
#   qat_int4.py 的 2:4 + INT8 QAT(tq_w8a8_sp24d_ft_qat_fold<k>.pth)目前只走 fake-quant 路徑;這裡照 qat_fx.py 對 dense INT8 的做法:
#   1. convert_fx 成整數模型 → 遮罩 tq_sp24d_int8(cv + test),和 fake-quant 版、dense 整數 INT8(tq_qat)配對;
#   2. 檢查整數權重是否仍是 2:4(每個 (輸出通道, kh, kw) 沿輸入通道每 4 個最多 2 個非 0;head.1 dense 不算);
#   3. qat_fx.to_fixed(32 位元累加器、16 位元乘數)→ 遮罩 tq_sp24d_fx(只跑測試集,同 tq_qat_fx)。


def check_24(qmodel):
    rows = []
    for name, m in qmodel.named_modules():
        if not hasattr(m, "weight") or not callable(m.weight):
            continue
        w = m.weight()
        if not hasattr(w, "int_repr") or w.dim() != 4:
            continue
        q = w.int_repr()
        o, i, kh, kw = q.shape
        nz = (q != 0).float().mean().item()
        ok = None
        if i % 4 == 0:
            g = q.permute(0, 2, 3, 1).reshape(-1, 4)
            ok = bool(((g != 0).sum(1) <= 2).all())
        rows.append((name, tuple(q.shape), round(nz, 4), ok))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    log = []
    for fold in args.folds:
        start = time.time()
        prepared = prepare(0, fold, 8, 8, "sp24d_ft").cpu()
        state = torch.load(OUT / "models" / f"tq_w8a8_sp24d_ft_qat_fold{fold}.pth", map_location="cpu")
        qmodel = convert(state, prepared)
        rows = check_24(qmodel)
        bad = [r for r in rows if r[3] is False]
        dense = [r[0] for r in rows if r[3] is None or r[2] > 0.55]
        print(f"[fold{fold}] 整數卷積 {len(rows)} 層;不符合 2:4 的層 {len(bad)} {[r[0] for r in bad]};非零比例 > 55% 或無法分組 {dense}", flush=True)
        log += [(fold, *r) for r in rows]
        sets = {"cv": read_split(f"fold{fold}_val")[:args.limit], "test": read_split("test")[:args.limit]}
        save_masks(qmodel, "tq_sp24d_int8", fold, sets)
        save_masks(to_fixed(qmodel, 32, 16), "tq_sp24d_fx", fold, {"test": sets["test"]})
        print(f"[fold{fold}] 完成 {time.time() - start:.0f} s", flush=True)
    with open(OUT / "sp24d_int8_check.csv", "w", newline="") as f:
        csv.writer(f).writerows([("fold", "layer", "shape", "nonzero", "is_24")] + log)


if __name__ == "__main__":
    main()
