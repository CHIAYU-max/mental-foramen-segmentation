"""2026-11-14 畫質(可辨識度)檢查與純影像數值的頦孔偵測(使用者要求;伺服器 CPU,探索性;輸出 compare_results/quality_gate/)

文獻裡處理看不清楚的影像的第二類做法: 在模型前面加一個「這張能不能判讀」的檢查。這裡測兩件事:
A. 可辨識度分類器 + 轉介: 能不能比現在的「模型分歧度 u」更會挑出沒命中的 ROI。
   預測與 u 同 pano_uncertainty.py --post lcc: CV = 同一折 5 個種子的多數決(559),測試 = 5 種子 × 5 折(137),lcc;u 不需要 GT。
   特徵(推論時都拿得到):
     I  影像統計(quality_errors.global_quality): 平均、標準差、p5–p95、清晰度、雜訊、尺寸
     Vf 可辨識度分類器(凍結): ImageNet ResNet34 的全域平均特徵 + logistic regression,目標 = 人工判定的 mf_visible
     Vt 可辨識度分類器(微調): ImageNet ResNet18 @ 384,固定 12 epoch、不挑 checkpoint
     M  模型輸出(種子 0 的機率圖;CV = 沒看過它的那一折,測試 = 五折各算一次再平均): 頦孔機率最大值、機率總量、區塊數、第二大 / 最大區塊、最大區塊內平均機率
     B  純影像數值(見 B 部分)在預測位置的暗斑反應、與純影像候選點的距離
   轉介分數: u 單獨;各組單獨(LR);u + 各組;全部。LR 在 CV 以折為單位 out-of-fold(折是以病人 ∪ 全景片分組),測試用全部 CV 訓練。
   指標: AUROC(沒命中)、AUROC(無法辨識)、保留 90% / 80% 的命中率;與 u 的差以組為單位 bootstrap 2000 次。
   事先定的判準: CV 保留 90% 的命中率比 u 高、95% CI 下界 > 0,且測試不變差,才建議換轉介分數。
B. 純影像數值找頦孔(不用神經網路): 頦孔是小的圓形暗斑 → 多尺度、尺度正規化的 LoG(暗斑為正),尺度由訓練折的 GT 半徑(10 / 30 / 50 / 70 / 90 百分位)決定;
   候選 = 反應的局部最大(抑制半徑 = 訓練折半徑中位數)。另用訓練折 GT 中心的位置分布(正規化座標 32 × 32 直方圖,高斯平滑)當位置先驗。
   三個版本: 只用位置先驗(最可能的位置,基準)、只用暗斑、暗斑(以影像內標準差正規化)+ log 位置先驗(權重固定 1)。
   命中 = 候選中心在 GT 中心 1R 內(與模型相同);報 top-1 / top-3 / top-5,分可辨識 / 無法辨識。
   和模型互補: (a) 模型沒有預測時改用純影像 top-1;(b) 多數決有多個區塊時改選暗斑反應最大的區塊(原本選面積最大);(c) 當轉介特徵(A 的 B 組)。
"""
import csv
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, ROOT, index_files, load_gt
from stage2.kfold_pano import EXCLUDE_TEST, PROJECT
from stage2.kfold_unet_variants import fold_paths
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.mf_postproc import largest
from analysis.paper_metrics import read_masks
from analysis.pano_uncertainty import iou
from analysis.pano_visibility import VISIBILITY_CSV
from stage2.quality_errors import TAGS, global_quality
from stage2.quality_methods import load_model, probs

MASKS = ROOT / "compare_results" / PROJECT
OUT = ROOT / "compare_results" / "quality_gate"
SIDE = re.compile(r"_(Left|Right)_roi$")
I_KEYS = ["mean", "std", "range_p5_p95", "sharpness", "noise", "height", "width"]
M_KEYS = ["p_max", "p_mass", "comp_n", "comp_ratio", "p_in"]
B_KEYS = ["blob_z", "blob_pct", "blob_dist"]


def load_items():
    visible = {r["stem"]: int(r["mf_visible"]) for r in csv.DictReader(open(VISIBILITY_CSV, newline=""))}
    folds = {r["stem"]: r for r in csv.DictReader(open(MASKS / "folds.csv", newline=""))}
    items = []
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", MASKS):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            gt = load_gt(path.stem, img.shape, path.parent.parent / "labels", {})["mf"]
            if gt.any():
                members = [read_masks(MASKS, "cv", t, fold, path.stem)["mf"] for t in TAGS]
                items.append(dict(setting="cv", fold=fold, stem=path.stem, group=folds[path.stem]["group"], img=img, gt=gt, members=members,
                                  visible=visible.get(path.stem)))
    test_gt = (ROOT / GT_DIR).resolve()
    for stem, path in sorted(index_files((ROOT / IMAGES_DIR).resolve()).items()):
        if stem in EXCLUDE_TEST:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gt = load_gt(stem, img.shape, test_gt, {})["mf"]
        if gt.any():
            members = [read_masks(MASKS, "test", t, f, stem)["mf"] for t in TAGS for f in range(K_FOLDS)]
            items.append(dict(setting="test", fold=None, stem=stem, group="t_" + SIDE.sub("", stem), img=img, gt=gt, members=members, visible=visible.get(stem)))
    for it in items:
        m = it["members"]
        vote = np.sum(m, axis=0) > len(m) / 2
        it["vote"], it["pred"] = vote, largest(vote)
        if not vote.any() and not any(x.any() for x in m):
            it["u"] = 0.0
        else:
            it["u"] = 1.0 - float(np.mean([iou(x, vote) if x.any() and vote.any() else 0.0 for x in m]))
        it["c"] = np.argwhere(it["gt"]).mean(0)
        it["R"] = (it["gt"].sum() / np.pi) ** 0.5
        it["hit"] = hit_mask(it["pred"], it)
    return items


def hit_point(p, it):
    return float(np.linalg.norm(np.asarray(p, float) - it["c"]) <= it["R"])


def hit_mask(mask, it):
    return hit_point(np.argwhere(mask).mean(0), it) if mask.any() else 0.0


# ---------- B. 純影像數值 ----------
def blob_map(img, sigmas):
    f = img.astype(np.float32)
    f = (f - f.mean()) / (f.std() + 1e-6)
    return np.max([s * s * cv2.Laplacian(cv2.GaussianBlur(f, (0, 0), s), cv2.CV_32F, ksize=3) for s in sigmas], axis=0)


def peaks(score, r, k=5):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    mx = cv2.dilate(score, ker)
    ys, xs = np.nonzero((score >= mx) & np.isfinite(score))
    order = np.argsort(-score[ys, xs])[:k]
    return [(ys[i], xs[i]) for i in order]


def prior_map(train, shape):
    h = np.zeros((32, 32), np.float32)
    for it in train:
        y, x = it["c"][0] / it["img"].shape[0], it["c"][1] / it["img"].shape[1]
        h[min(31, int(y * 32)), min(31, int(x * 32))] += 1
    h = cv2.GaussianBlur(h, (0, 0), 2)
    h /= h.max()
    return cv2.resize(h, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def classical(it, train):
    radii = np.array([t["R"] for t in train])
    sigmas = np.percentile(radii, [10, 30, 50, 70, 90]) / 2 ** 0.5
    r = int(round(np.median(radii)))
    resp = blob_map(it["img"], sigmas)
    z = (resp - resp.mean()) / (resp.std() + 1e-6)
    prior = prior_map(train, resp.shape)
    scores = {"prior": prior, "blob": z, "blob+prior": z + np.log(prior + 1e-6)}
    out = {}
    for name, s in scores.items():
        pk = peaks(s.astype(np.float32), r)
        for k in (1, 3, 5):
            out[f"{name}|top{k}"] = float(any(hit_point(p, it) for p in pk[:k]))
        out[f"{name}|peaks"] = pk
    out["z"], out["r"] = z, r
    return out


# ---------- A. 特徵 ----------
def prob_feats(p):
    mf, arg = p[1], p.argmax(0) == 1
    n, lab, stats, _ = cv2.connectedComponentsWithStats(arg.astype(np.uint8), 8)
    if n < 2:
        return dict(p_max=float(mf.max()), p_mass=float(np.log1p(mf.sum())), comp_n=0.0, comp_ratio=0.0, p_in=0.0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    return dict(p_max=float(mf.max()), p_mass=float(np.log1p(mf.sum())), comp_n=float(n - 1),
                comp_ratio=float(areas[order[1]] / areas[order[0]]) if n > 2 else 0.0, p_in=float(mf[lab == order[0] + 1].mean()))


def blob_feats(it, cl):
    if not it["pred"].any():
        return dict(blob_z=0.0, blob_pct=0.0, blob_dist=10.0)
    z = cl["z"]
    v = float(z[it["pred"]].max())
    c = np.argwhere(it["pred"]).mean(0)
    rp = (it["pred"].sum() / np.pi) ** 0.5
    top = np.asarray(cl["blob+prior|peaks"][0], float)
    return dict(blob_z=v, blob_pct=float((z < v).mean()), blob_dist=float(min(10.0, np.linalg.norm(c - top) / max(rp, 1.0))))


def imagenet_feats(items):
    import torchvision
    net = torchvision.models.resnet34(weights="IMAGENET1K_V1").eval()
    net.fc = torch.nn.Identity()
    feats = []
    with torch.inference_mode():
        for it in items:
            x = cv2.resize(it["img"], (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            x = torch.from_numpy((x - 0.449) / 0.226)[None, None].repeat(1, 3, 1, 1)
            feats.append(net(x)[0].numpy())
    return np.stack(feats)


def finetune_vis(items):
    """ResNet18 微調的可辨識度分類器: CV out-of-fold;測試 = 五個折模型平均。回傳 P(可辨識)。"""
    cache = OUT / "vis_cnn.csv"
    if cache.exists():
        d = {r["stem"]: float(r["p_visible"]) for r in csv.DictReader(open(cache, newline=""))}
        return np.array([d[it["stem"]] for it in items])
    import torchvision
    torch.manual_seed(0)
    X = torch.from_numpy(np.stack([(cv2.resize(it["img"], (384, 384), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - 0.449) / 0.226 for it in items]))[:, None]
    y = torch.tensor([float(it["visible"]) if it["visible"] is not None else -1.0 for it in items])
    fold = np.array([-1 if it["fold"] is None else it["fold"] for it in items])
    out = np.zeros(len(items))

    def augment(xb):
        n = len(xb)
        ang = (torch.rand(n) - 0.5) * 2 * np.deg2rad(10)
        sc = 0.9 + 0.2 * torch.rand(n)
        flip = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
        theta = torch.zeros(n, 2, 3)
        theta[:, 0, 0], theta[:, 0, 1] = torch.cos(ang) / sc * flip, -torch.sin(ang) / sc
        theta[:, 1, 0], theta[:, 1, 1] = torch.sin(ang) / sc * flip, torch.cos(ang) / sc
        theta[:, :, 2] = (torch.rand(n, 2) - 0.5) * 0.1
        xb = torch.nn.functional.grid_sample(xb, torch.nn.functional.affine_grid(theta, xb.shape, align_corners=False), padding_mode="border", align_corners=False)
        return xb * (1 + 0.3 * (torch.rand(n, 1, 1, 1) - 0.5)) + 0.3 * (torch.rand(n, 1, 1, 1) - 0.5)

    def predict(net, idx):
        net.eval()
        ps = []
        with torch.inference_mode():
            for i in range(0, len(idx), 32):
                xb = X[idx[i:i + 32]].repeat(1, 3, 1, 1)
                ps.append(((torch.sigmoid(net(xb)) + torch.sigmoid(net(xb.flip(-1)))) / 2)[:, 0])
        return torch.cat(ps).numpy()

    test_idx = np.nonzero(fold < 0)[0]
    for k in range(K_FOLDS):
        tr = np.nonzero((fold >= 0) & (fold != k) & (y.numpy() >= 0))[0]
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        net.fc = torch.nn.Linear(512, 1)
        epochs, bs = 12, 16
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(tr) // bs))
        for ep in range(epochs):
            net.train()
            perm = np.random.default_rng(100 * k + ep).permutation(tr)
            tot = 0.0
            for i in range(0, len(perm) - bs + 1, bs):
                b = perm[i:i + bs]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(augment(X[b]).repeat(1, 3, 1, 1))[:, 0], y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item()
            print(f"  可辨識度微調 fold {k} epoch {ep + 1}/{epochs} loss {tot / (len(perm) // bs):.3f}", flush=True)
        val = np.nonzero(fold == k)[0]
        out[val] = predict(net, val)
        out[test_idx] += predict(net, test_idx) / K_FOLDS
    with open(cache, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "p_visible"])
        w.writerows([it["stem"], f"{p:.5f}"] for it, p in zip(items, out))
    return out


# ---------- 統計 ----------
def auroc(score, label):
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label)
    return float(roc_auc_score(label, score)) if 0 < label.sum() < len(label) else float("nan")


def kept_hit(score, hit, cover):
    order = np.argsort(score, kind="stable")
    return float(np.mean(hit[order[:max(1, int(round(cover * len(hit))))]]))


def boot_diff(fn, groups, n=2000, seed=0):
    keys = sorted(set(groups))
    members = {k: np.nonzero(np.array(groups) == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        v = fn(idx)
        if v == v:
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def lr_scores(Xcv, ycv, fold, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    cv = np.zeros(len(ycv))
    for k in range(K_FOLDS):
        ok = (fold != k) & (ycv >= 0)
        cv[fold == k] = mk().fit(Xcv[ok], ycv[ok]).predict_proba(Xcv[fold == k])[:, 1]
    ok = ycv >= 0
    return cv, mk().fit(Xcv[ok], ycv[ok]).predict_proba(Xte)[:, 1]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(24)
    items = load_items()
    cv_items = [it for it in items if it["setting"] == "cv"]
    te_items = [it for it in items if it["setting"] == "test"]
    print(f"CV {len(cv_items)}、測試 {len(te_items)};沒命中 CV {sum(1 - it['hit'] for it in cv_items):.0f}、測試 {sum(1 - it['hit'] for it in te_items):.0f}", flush=True)

    # B. 純影像數值
    lines = ["# 畫質檢查與純影像數值的頦孔偵測(quality_gate.py,2026-11-14,探索性)\n", "## B. 純影像數值找頦孔\n",
             "| 資料 | 分層 | n | 模型(多數決 lcc) | 只用位置先驗 top-1 | 暗斑 top-1 / 3 / 5 | 暗斑 + 位置 top-1 / 3 / 5 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for it in items:
        train = [t for t in cv_items if t["fold"] != it["fold"]] if it["setting"] == "cv" else cv_items
        it["cl"] = classical(it, train)
        it.update(blob_feats(it, it["cl"]))
        it.update(global_quality(it["img"]))
    crow = []
    for setting, sub in (("cv", cv_items), ("test", te_items)):
        for name, sel in (("全部", lambda t: True), ("可辨識", lambda t: t["visible"] == 1), ("無法辨識", lambda t: t["visible"] == 0)):
            s = [t for t in sub if sel(t)]
            m = lambda key: np.mean([t["cl"][key] for t in s])
            lines.append(f"| {setting} | {name} | {len(s)} | {np.mean([t['hit'] for t in s]):.3f} | {m('prior|top1'):.3f} | "
                         f"{m('blob|top1'):.3f} / {m('blob|top3'):.3f} / {m('blob|top5'):.3f} | {m('blob+prior|top1'):.3f} / {m('blob+prior|top3'):.3f} / {m('blob+prior|top5'):.3f} |")
            crow.append({"setting": setting, "stratum": name, "n": len(s), "model_hit": np.mean([t["hit"] for t in s]),
                         **{k: m(k) for k in ("prior|top1", "blob|top1", "blob|top3", "blob|top5", "blob+prior|top1", "blob+prior|top3", "blob+prior|top5")}})
    # 互補
    lines += ["", "### 和模型互補", ""]
    for setting, sub in (("cv", cv_items), ("test", te_items)):
        groups = [t["group"] for t in sub]
        base = np.array([t["hit"] for t in sub])
        fb, rr = base.copy(), base.copy()
        n_empty = n_multi = 0
        for i, t in enumerate(sub):
            if not t["pred"].any():
                n_empty += 1
                fb[i] = hit_point(t["cl"]["blob+prior|peaks"][0], t)
            n, lab = cv2.connectedComponents(t["vote"].astype(np.uint8), connectivity=8)
            if n > 2:
                n_multi += 1
                best = max(range(1, n), key=lambda j: t["cl"]["z"][lab == j].max())
                rr[i] = hit_mask(lab == best, t)
        for name, new, cnt in (("(a) 沒有預測時改用純影像 top-1", fb, n_empty), ("(b) 多個區塊時選暗斑反應最大的", rr, n_multi)):
            lo, hi = boot_diff(lambda idx: float(np.mean(new[idx] - base[idx])), groups)
            lines.append(f"- {setting} {name}: 影響 {cnt} 個 ROI,命中 {base.mean():.4f} → {new.mean():.4f}(差 {100 * (new.mean() - base.mean()):+.2f} 點 [{100 * lo:+.2f}, {100 * hi:+.2f}];得 {int(((new > base)).sum())} / 失 {int((new < base).sum())})")
            crow.append({"setting": setting, "stratum": name, "n": cnt, "model_hit": base.mean(), "new_hit": new.mean(), "lo": lo, "hi": hi})
    lines += ["", "### 純影像和模型各自對在哪裡", ""]
    for setting, sub in (("cv", cv_items), ("test", te_items)):
        h = np.array([t["hit"] for t in sub])
        c1 = np.array([t["cl"]["blob+prior|top1"] for t in sub])
        c3 = np.array([t["cl"]["blob+prior|top3"] for t in sub])
        agree = np.array([t["blob_dist"] <= 1.0 for t in sub])
        lines.append(f"- {setting}: 模型沒命中 {int((h == 0).sum())} 個,其中純影像 top-1 命中 {int(c1[h == 0].sum())}、top-3 命中 {int(c3[h == 0].sum())};"
                     f"「模型或純影像 top-1 任一命中」的上限 {np.mean((h + c1) > 0):.4f}(模型 {h.mean():.4f});模型命中的 ROI 裡純影像 top-1 命中 {c1[h == 1].mean():.3f}。"
                     f"模型預測在純影像 top-1 的 1 個預測半徑內 {agree.mean():.1%}: 一致時模型命中 {h[agree].mean():.4f}(n = {int(agree.sum())}),不一致時 {h[~agree].mean():.4f}(n = {int((~agree).sum())})")
    with open(OUT / "classical.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in crow for k in r)))
        w.writeheader()
        w.writerows(crow)
    print("\n".join(lines), flush=True)

    # A. 特徵
    models = {k: load_model(k) for k in range(K_FOLDS)}
    for i, it in enumerate(items):
        ps = [prob_feats(probs(models[k], it["img"])) for k in ([it["fold"]] if it["setting"] == "cv" else range(K_FOLDS))]
        it.update({k: float(np.mean([p[k] for p in ps])) for k in M_KEYS})
        if i % 100 == 0:
            print(f"模型機率特徵 {i}/{len(items)}", flush=True)
    del models
    fold = np.array([it["fold"] for it in cv_items])
    ncv = len(cv_items)
    vis = np.array([-1 if it["visible"] is None else it["visible"] for it in items])
    F = imagenet_feats(items)
    from sklearn.decomposition import PCA
    Fp = PCA(32, random_state=0).fit(F[:ncv]).transform(F)  # 無標註的降維,只用 CV 影像擬合
    vf_cv, vf_te = lr_scores(Fp[:ncv], vis[:ncv], fold, Fp[ncv:])
    vt = finetune_vis(items)
    for it, a, b in zip(items, np.concatenate([vf_cv, vf_te]), vt):
        it["Vf"], it["Vt"] = 1 - a, 1 - b  # 分數 = 無法辨識的機率

    miss = np.array([1 - it["hit"] for it in items])
    hit = 1 - miss
    col = lambda keys: np.array([[it[k] for k in keys] for it in items], float)
    sets = {"I 影像統計": I_KEYS, "Vf 可辨識度(凍結)": ["Vf"], "Vt 可辨識度(微調)": ["Vt"], "M 模型輸出": M_KEYS, "B 純影像暗斑": B_KEYS,
            "u + Vt": ["u", "Vt"], "u + M": ["u"] + M_KEYS, "u + B": ["u"] + B_KEYS, "u + I": ["u"] + I_KEYS,
            "全部(u + I + Vt + M + B)": ["u"] + I_KEYS + ["Vt"] + M_KEYS + B_KEYS}
    scores = {"u 模型分歧度(現行)": np.array([it["u"] for it in items])}
    for name, keys in sets.items():
        X = col(keys)
        if len(keys) == 1:
            scores[name] = X[:, 0]
        else:
            a, b = lr_scores(X[:ncv], miss[:ncv], fold, X[ncv:])
            scores[name] = np.concatenate([a, b])
    # 可辨識度本身分得出來嗎
    lines += ["", "## A. 可辨識度檢查與轉介", "", "### 分數能不能分出「無法辨識」(AUROC)", "", "| 分數 | CV | 測試 |", "| --- | --- | --- |"]
    Xi = col(I_KEYS)
    a, b = lr_scores(Xi[:ncv], np.where(vis[:ncv] >= 0, 1 - vis[:ncv], -1), fold, Xi[ncv:])
    vis_scores = {"u 模型分歧度": scores["u 模型分歧度(現行)"], "I 影像統計(LR)": np.concatenate([a, b]), "Vf 可辨識度(凍結)": scores["Vf 可辨識度(凍結)"], "Vt 可辨識度(微調)": scores["Vt 可辨識度(微調)"]}
    rows = []
    for name, s in vis_scores.items():
        okc, okt = vis[:ncv] >= 0, vis[ncv:] >= 0
        ac, at = auroc(s[:ncv][okc], 1 - vis[:ncv][okc]), auroc(s[ncv:][okt], 1 - vis[ncv:][okt])
        lines.append(f"| {name} | {ac:.3f} | {at:.3f} |")
        rows.append({"table": "not_visible", "score": name, "cv_auroc": ac, "test_auroc": at})
    lines += ["", "### 轉介分數(挑出沒命中的 ROI)", "", "| 轉介分數 | CV AUROC | CV 保留 90% 命中 | 與 u 的差 [95% CI] | CV 保留 80% | 測試 AUROC | 測試 保留 90% | 與 u 的差 [95% CI] | 測試 保留 80% |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    u = scores["u 模型分歧度(現行)"]
    for name, s in scores.items():
        r = {"table": "referral", "score": name}
        cells = []
        for tag, sl in (("cv", slice(0, ncv)), ("test", slice(ncv, None))):
            ss, uu, hh, mm = s[sl], u[sl], hit[sl], miss[sl]
            groups = [it["group"] for it in items[sl]]
            k90, k80 = kept_hit(ss, hh, 0.9), kept_hit(ss, hh, 0.8)
            d = k90 - kept_hit(uu, hh, 0.9)
            lo, hi = boot_diff(lambda idx: kept_hit(ss[idx], hh[idx], 0.9) - kept_hit(uu[idx], hh[idx], 0.9), groups) if name != "u 模型分歧度(現行)" else (0.0, 0.0)
            r.update({f"{tag}_auroc": auroc(ss, mm), f"{tag}_hit90": k90, f"{tag}_hit90_diff": d, f"{tag}_lo": lo, f"{tag}_hi": hi, f"{tag}_hit80": k80})
            cells += [f"{auroc(ss, mm):.3f}", f"{k90:.3f}", "—" if name.startswith("u 模型") else f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]", f"{k80:.3f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
        rows.append(r)
    with open(OUT / "gate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    keys = ["setting", "stem", "group", "visible", "hit", "u", "Vf", "Vt"] + I_KEYS + M_KEYS + B_KEYS
    with open(OUT / "per_roi.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["prior_top1", "blob_top1", "blobprior_top1", "blobprior_top3"])
        w.writeheader()
        for it in items:
            w.writerow({**{k: it[k] for k in keys}, "prior_top1": it["cl"]["prior|top1"], "blob_top1": it["cl"]["blob|top1"],
                        "blobprior_top1": it["cl"]["blob+prior|top1"], "blobprior_top3": it["cl"]["blob+prior|top3"]})
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
