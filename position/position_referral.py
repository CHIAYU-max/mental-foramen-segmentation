# 2026-11-05 實驗 1: 系統層級的「轉介率 vs 準確率」(只讀,輸出在 compare_results/mf_position/referral/)
# 對象: 測試集全部已標註的側(正常牙 + 缺牙 + FDI 錯 + 其他),主結果 R34/512 FP32 五折、5 種子(position_seeds.py 的 per_seed.csv)。
# 每側結果: 自動答對 / 自動答錯(沒被標記的錯 = 靜默錯誤)/ 轉介給人。異常側(缺牙、FDI 錯、其他)只要沒被轉介就算錯。
#   沒有預測頦孔或參考線不完整 → 系統本來就給不出答案 → 算轉介。
# 事先固定的策略(不看標註,轉介比例在測試集上以排名決定):
#   P0 不轉介;P1 偵測器(兩種編號不一致 或 頦孔側比對側少 ≥ 1 顆,11/4 定案);
#   P2 = P1 + 分歧度 u 最高 10%;P3 = P1 + 邊界距離 margin 最小 10%;P4 = P1 + 兩者。
#   u: uncertainty_per_roi_lcc.csv 測試集(5 種子 × 5 折 = 25 個成員);margin: 預測頦孔質心到最近判定邊界 / 頦孔半寬(各種子)。
# 正確與否以原始標註為主(SOP 用),另報複查後;CI 以全景片 bootstrap 2000 次。
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "YOLOv8"
MP = ROOT / "compare_results" / "mf_position"
DST = MP / "referral"
MAIN = "R34/512 FP32 五折(主結果)"
RATE = 0.10


def load():
    key = {r["alias"]: r for r in csv.DictReader(open(ROOT / "mf_position" / "key.csv", newline=""))}
    seeds = defaultdict(dict)
    for r in csv.DictReader(open(MP / "position_seeds" / "per_seed.csv", newline="")):
        if r["variant"] == MAIN:
            seeds[int(r["seed"])][r["alias"]] = r
    orig = {r["alias"]: r for r in csv.DictReader(open(ROOT / "mf_position" / "labels_rater.csv", newline="", encoding="utf-8-sig"))}
    adj = {r["alias"]: r for r in csv.DictReader(open(MP / "labels_adjudicated_dir" / "labels_rater.csv", newline="", encoding="utf-8-sig"))}
    fdi = {r["alias"]: int(r["flag_x456"]) for r in csv.DictReader(open(MP / "fdi_flag" / "per_side.csv", newline=""))}
    side = {r["alias"]: int(float(r["diff"]) >= 1) for r in csv.DictReader(open(MP / "side_count" / "per_side.csv", newline=""))}
    u = {r["stem"]: float(r["u"]) for r in csv.DictReader(open(ROOT / "compare_results/kfold_pano_mf_559/uncertainty_per_roi_lcc.csv", newline="")) if r["setting"] == "test"}
    return key, seeds, orig, adj, fdi, side, u


def group(a):
    if a["fdi_wrong"] == "1":
        return "FDI 錯"
    if a["position"] == "M":
        return "缺牙"
    if a["position"] in tuple("123456"):
        return "正常牙"
    return "其他"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    key, seeds, orig, adj, fdi, side, u = load()
    aliases = sorted(a for a in seeds[0] if adj.get(a, {}).get("position") or adj.get(a, {}).get("fdi_wrong") == "1")
    grp = {a: group(adj[a]) for a in aliases}
    pano = {a: key[a]["pano"] for a in aliases}
    det = {a: int(fdi.get(a, 0) or side.get(a, 0)) for a in aliases}
    uu = np.array([u[key[a]["stem"]] for a in aliases])
    n = len(aliases)
    k = int(round(RATE * n))
    u_top = set(np.array(aliases)[np.argsort(-uu)[:k]])
    print(f"側數 {n}:", dict(sorted({g: sum(1 for a in aliases if grp[a] == g) for g in set(grp.values())}.items())))

    def outcome(policy, s, lab):
        # 回傳 {alias: 'auto_ok' / 'auto_wrong' / 'refer'}
        rows = seeds[s]
        mg = {a: float(rows[a]["margin"]) for a in aliases if rows[a]["margin"] != ""}
        m_low = set(sorted(mg, key=mg.get)[:k])
        out = {}
        for a in aliases:
            r = rows[a]
            refer = r["pred"] == ""
            if policy >= 1:
                refer |= bool(det[a])
            if policy in (2, 4):
                refer |= a in u_top
            if policy in (3, 4):
                refer |= a in m_low
            if refer:
                out[a] = "refer"
            else:
                truth = (orig if lab == "orig" else adj)[a]["position"]
                ok = grp[a] == "正常牙" and r["pred"] == truth
                out[a] = "auto_ok" if ok else "auto_wrong"
        return out

    def boot_ci(vals_by_alias, denom_by_alias):
        panos = sorted(set(pano.values()))
        num = defaultdict(float); den = defaultdict(float)
        for a in aliases:
            num[pano[a]] += vals_by_alias[a]; den[pano[a]] += denom_by_alias[a]
        N = np.array([num[p] for p in panos]); D = np.array([den[p] for p in panos])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(panos), (2000, len(panos)))
        st = N[idx].sum(1) / np.maximum(D[idx].sum(1), 1e-9)
        return np.percentile(st, [2.5, 97.5])

    names = {0: "P0 不轉介", 1: "P1 偵測器(現行)", 2: "P2 P1 + 分歧度最高 10%", 3: "P3 P1 + 邊界最近 10%", 4: "P4 P1 + 兩者"}
    lines = []
    table = []  # summary.csv(build_results_xlsx.py 的「牙位」工作表讀這份)
    for lab in ("orig", "adj"):
        lines.append(f"\n=== 正確與否: {'原始' if lab == 'orig' else '複查後'}標註;5 種子平均(n = {n} 側)===")
        lines.append("| 策略 | 轉介 | 自動答對 | 靜默錯誤(錯且沒轉介) | 自動回答中的正確率 [95% CI] | 異常側被轉介 | 正常牙被轉介 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for p in range(5):
            outs = [outcome(p, s, lab) for s in sorted(seeds)]
            f = lambda a, v: np.mean([o[a] == v for o in outs])
            ok = {a: f(a, "auto_ok") for a in aliases}
            wr = {a: f(a, "auto_wrong") for a in aliases}
            rf = {a: f(a, "refer") for a in aliases}
            ans = {a: ok[a] + wr[a] for a in aliases}
            lo, hi = boot_ci(ok, ans)
            ab = [a for a in aliases if grp[a] != "正常牙"]
            nm = [a for a in aliases if grp[a] == "正常牙"]
            table.append({"labels": lab, "policy": names[p], "sides": n, "referred": round(sum(rf.values()), 2),
                          "auto_ok": round(sum(ok.values()), 2), "silent_wrong": round(sum(wr.values()), 2),
                          "auto_accuracy": round(sum(ok.values()) / sum(ans.values()), 5), "ci_lo": round(float(lo), 5), "ci_hi": round(float(hi), 5),
                          "abnormal_referred": round(sum(rf[a] for a in ab), 2), "abnormal": len(ab),
                          "normal_referred": round(sum(rf[a] for a in nm), 2), "normal": len(nm)})
            lines.append(f"| {names[p]} | {sum(rf.values()):.1f}({100 * np.mean(list(rf.values())):.1f}%) | {sum(ok.values()):.1f} | "
                         f"{sum(wr.values()):.1f}({100 * np.mean(list(wr.values())):.1f}%) | "
                         f"{100 * sum(ok.values()) / sum(ans.values()):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] | "
                         f"{sum(rf[a] for a in ab):.1f} / {len(ab)} | {sum(rf[a] for a in nm):.1f} / {len(nm)} |")
    # 曲線: P1 之後再依分數多轉介(原始標註)
    curves = {}
    base = [outcome(1, s, "orig") for s in sorted(seeds)]
    for score in ("分歧度 u", "邊界距離", "兩者取較危險"):
        xs, ys, zs = [], [], []
        for extra in range(0, n // 2 + 1):
            accs, refs, sil = [], [], []
            for s, o in zip(sorted(seeds), base):
                rows = seeds[s]
                left = [a for a in aliases if o[a] != "refer"]
                ur = {a: u[key[a]["stem"]] for a in left}
                mr = {a: -float(rows[a]["margin"]) for a in left}
                rank = lambda d: {a: i / max(len(d) - 1, 1) for i, a in enumerate(sorted(d, key=d.get))}
                ru, rm = rank(ur), rank(mr)
                sc = ur if score == "分歧度 u" else mr if score == "邊界距離" else {a: max(ru[a], rm[a]) for a in left}
                drop = set(sorted(left, key=lambda a: -sc[a])[:extra])
                kept = [a for a in left if a not in drop]
                accs.append(np.mean([o[a] == "auto_ok" for a in kept]) if kept else np.nan)
                refs.append(1 - len(kept) / n)
                sil.append(sum(o[a] == "auto_wrong" for a in kept) / n)
            xs.append(np.mean(refs)); ys.append(np.nanmean(accs)); zs.append(np.mean(sil))
        curves[score] = (xs, ys, zs)
    with open(DST / "curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "referral_rate", "auto_accuracy", "silent_error_rate"])
        for sname, (xs, ys, zs) in curves.items():
            for x, y, z in zip(xs, ys, zs):
                w.writerow([sname, round(x, 4), round(y, 4), round(z, 4)])
    for target in (0.25, 0.30, 0.35):
        row = []
        for sname, (xs, ys, zs) in curves.items():
            i = int(np.argmin(np.abs(np.array(xs) - target)))
            row.append(f"{sname}: 正確率 {100 * ys[i]:.1f}%、靜默錯誤 {100 * zs[i]:.1f}%(轉介 {100 * xs[i]:.1f}%)")
        lines.append(f"\n總轉介約 {int(100 * target)}%: " + ";".join(row))
    text = "\n".join(lines)
    print(text)
    (DST / "summary.md").write_text(text + "\n", encoding="utf-8")
    with open(DST / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    # 圖: 單一 y 軸(自動回答中的正確率),三條線 + 不轉介 / 偵測器兩個點
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    colors = {"分歧度 u": "#2a78d6", "邊界距離": "#eb6834", "兩者取較危險": "#1baf7a"}
    labels_en = {"分歧度 u": "model disagreement u", "邊界距離": "distance to position boundary", "兩者取較危險": "either (max rank)"}
    for sname, (xs, ys, _) in curves.items():
        ax.plot(np.array(xs) * 100, np.array(ys) * 100, color=colors[sname], lw=2, label=labels_en[sname])
    p0 = [outcome(0, s, "orig") for s in sorted(seeds)]
    acc0 = np.mean([np.mean([o[a] == "auto_ok" for a in aliases if o[a] != "refer"]) for o in p0])
    ref0 = np.mean([np.mean([o[a] == "refer" for a in aliases]) for o in p0])
    ax.scatter([ref0 * 100], [acc0 * 100], s=40, color="#52514e", zorder=3)
    ax.annotate("no referral", (ref0 * 100, acc0 * 100), xytext=(6, -10), textcoords="offset points", fontsize=8, color="#52514e")
    xs, ys, _ = curves["分歧度 u"]
    ax.scatter([xs[0] * 100], [ys[0] * 100], s=40, color="#0b0b0b", zorder=3)
    ax.annotate("current detector (P1)", (xs[0] * 100, ys[0] * 100), xytext=(6, -12), textcoords="offset points", fontsize=8, color="#0b0b0b")
    ax.set_xlabel("sides referred to a clinician (%)", color="#52514e")
    ax.set_ylabel("accuracy of automatic answers (%)", color="#52514e")
    ax.set_title("Tooth position: referral vs accuracy (test set, all sides, 5 seeds)", fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(True, color="#e6e5e0", lw=0.6); [s.set_visible(False) for s in ax.spines.values()]
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_ylim(70, 100)
    ax.set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(DST / "referral_curve.png")
    print("圖:", DST / "referral_curve.png")



if __name__ == "__main__":
    main()
