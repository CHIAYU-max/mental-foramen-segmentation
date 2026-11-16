"""Regenerate the README figures in docs/figures.

The numbers are aggregate results copied from the experiment outputs. No image, mask or per-patient value is read.
Run from the repository root: python docs/make_figures.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures"

STYLE = """<style>
.bg{fill:#fcfcfb}.t{fill:#0b0b0b}.s{fill:#52514e}.grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}.edge{stroke:#8a8983}
.a{--c:#2a78d6}.b{--c:#eb6834}.c{--c:#6e6d68}
.fill{fill:var(--c)}.line{stroke:var(--c)}.ring{stroke:#fcfcfb}.tint{fill:var(--c);fill-opacity:.12}.card{fill:#fcfcfb}
text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;font-size:13px}
.title{font-size:16px;font-weight:600}.small{font-size:12px}.bold{font-weight:600}
@media (prefers-color-scheme:dark){
.bg,.card{fill:#1a1a19}.t{fill:#f0efec}.s{fill:#c3c2b7}.grid{stroke:#2c2c2a}.axis{stroke:#383835}.edge{stroke:#8a8983}
.a{--c:#3987e5}.b{--c:#d95926}.c{--c:#a9a89f}.ring{stroke:#1a1a19}}
</style>"""


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width, height, title, subtitle, label):
        self.width, self.height = width, height
        self.parts = [f'<rect class="bg" width="{width}" height="{height}" rx="8"/>']
        self.label = label
        self.text(24, 30, title, "t title")
        if subtitle:
            self.text(24, 50, subtitle, "s small")

    def text(self, x, y, value, cls="t", anchor="start"):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{esc(value)}</text>')

    def line(self, x1, y1, x2, y2, cls="grid", width=1, extra=""):
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" stroke-width="{width}" {extra}/>')

    def path(self, d, cls, width=2, extra=""):
        self.parts.append(f'<path d="{d}" class="{cls}" fill="none" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" {extra}/>')

    def dot(self, x, y, group, r=5):
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{group} fill ring" stroke-width="2"/>')

    def legend(self, x, y, items):
        for name, group in items:
            self.parts.append(f'<rect x="{x}" y="{y - 10}" width="11" height="11" rx="2" class="{group} fill"/>')
            self.text(x + 16, y, name, "s small")
            x += 24 + len(name) * 7

    def box(self, x, y, w, h, title, lines, group=None):
        cls = f'{group} tint line' if group else "card edge"
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}" stroke-width="{2 if group else 1.25}"/>')
        self.text(x + 14, y + 23, title, "t bold")
        for i, line in enumerate(lines):
            self.text(x + 14, y + 42 + i * 17, line, "s small")

    def arrow(self, x1, y1, x2, y2):
        self.line(x1, y1, x2, y2, "edge", 1.25, 'marker-end="url(#arrow)"')

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" role="img" aria-label="{esc(self.label)}">'
                f'{STYLE}<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                '<path d="M0 0L10 5L0 10z" fill="#8a8983"/></marker></defs>')
        (OUT / f"{name}.svg").write_text(head + "\n".join(self.parts) + "</svg>\n", encoding="utf-8", newline="\n")


def panel(svg, x0, y0, w, h, xs, ys, xticks, yticks, xfmt, yfmt, title=None):
    x = lambda v: x0 + (v - xs[0]) / (xs[1] - xs[0]) * w
    y = lambda v: y0 + h - (v - ys[0]) / (ys[1] - ys[0]) * h
    if title:
        svg.text(x0, y0 - 12, title, "t bold")
    for t in yticks:
        svg.line(x0, y(t), x0 + w, y(t))
        svg.text(x0 - 8, y(t) + 4, yfmt(t), "s small", "end")
    for t in xticks:
        svg.text(x(t), y0 + h + 19, xfmt(t), "s small", "middle")
    svg.line(x0, y0 + h, x0 + w, y0 + h, "axis")
    return x, y


def series(svg, x, y, points, group):
    svg.path("M" + "L".join(f"{x(a):.1f},{y(b):.1f}" for a, b in points), f"{group} line")
    for a, b in points:
        svg.dot(x(a), y(b), group)


def pipeline():
    s = Svg(860, 388, "Two-stage pipeline", "From a panoramic radiograph to the mental foramen, the nerve canal and the tooth position", "Two-stage segmentation pipeline")
    s.box(24, 74, 190, 74, "Panoramic radiograph", ["Input image"])
    s.box(254, 74, 270, 74, "Stage 1: teeth", ["YOLOv8s-seg, imgsz 1024", "stage1/fdi_cropped.py"], "a")
    s.box(564, 74, 272, 74, "FDI numbering", ["Filtering, jaw split, tooth numbers", "stage1/fdi_numbering.py"])
    s.box(564, 182, 272, 74, "ROI crop and unsharp mask", ["Teeth 34–36 and 44–46", "data/mf_unsharp_mask.py"])
    s.box(254, 182, 270, 74, "Stage 2: segmentation", ["U-Net with ResNet34 encoder, 512 × 512", "stage2/kfold_unet_variants.py"], "a")
    s.box(24, 182, 190, 74, "Five-fold ensemble", ["Vote; deployed as INT8 cascade", "deployment/deploy_pipeline.py"])
    s.box(24, 290, 190, 74, "Largest region (lcc)", ["One foramen per ROI", "stage2/mf_postproc.py"])
    s.box(254, 290, 270, 74, "Foramen and tooth position", ["Centroid, canal mask, FDI tooth position", "position/mf_position.py"], "b")
    s.box(564, 290, 272, 74, "Clinician review", ["Numbering flags, boundary margin", "position/position_referral.py"])
    s.arrow(214, 111, 254, 111)
    s.arrow(524, 111, 564, 111)
    s.arrow(700, 148, 700, 182)
    s.arrow(564, 219, 524, 219)
    s.arrow(254, 219, 214, 219)
    s.arrow(119, 256, 119, 290)
    s.arrow(214, 327, 254, 327)
    s.arrow(524, 327, 564, 327)
    s.save("pipeline")


def hit_threshold():
    s = Svg(860, 372, "The U-Net advantage comes from fewer large errors",
            "Foramen hit rate by distance threshold, five seeds, lcc. The test gap at 0.5 R is not significant (p = 0.21)", "Hit rate against threshold for U-Net and YOLO26s")
    s.legend(24, 74, [("U-Net (ResNet34)", "a"), ("YOLO26s", "b")])
    data = {"Cross-validation (559 ROIs)": ([(.5, 73.3), (1, 86.7), (1.5, 89.6), (2, 90.7)], [(.5, 71.0), (1, 82.9), (1.5, 86.1), (2, 87.3)]),
            "Test set (137 ROIs)": ([(.5, 75.6), (1, 93.0), (1.5, 94.3), (2, 94.5)], [(.5, 72.7), (1, 87.7), (1.5, 90.5), (2, 91.4)])}
    for i, (title, (unet, yolo)) in enumerate(data.items()):
        x, y = panel(s, 68 + i * 410, 118, 340, 196, (.4, 2.1), (60, 100), [.5, 1, 1.5, 2], [60, 70, 80, 90, 100], lambda t: f"{t:g} R", lambda t: f"{t}%", title)
        series(s, x, y, yolo, "b")
        series(s, x, y, unet, "a")
        s.text(x(2) - 2, y(unet[-1][1]) - 10, f"{unet[-1][1]}%", "t small", "end")
        s.text(x(2) - 2, y(yolo[-1][1]) + 18, f"{yolo[-1][1]}%", "t small", "end")
    s.text(430, 356, "Centroid error threshold, in equivalent radii of the ground-truth foramen", "s small", "middle")
    s.save("hit_threshold")


def referral():
    s = Svg(860, 372, "Referring uncertain cases raises the hit rate on the rest",
            "Ensemble disagreement decides which ROIs go to a clinician. Gain at 10% referred: CV +6.2 points (4.6–8.2), test +4.9 (1.4–7.8)", "Hit rate of retained ROIs by share referred")
    s.legend(24, 74, [("Test set (137 ROIs)", "a"), ("Cross-validation (559 ROIs)", "b")])
    x, y = panel(s, 68, 100, 760, 214, (-1, 21), (85, 100), [0, 5, 10, 20], [85, 90, 95, 100], lambda t: f"{t}%", lambda t: f"{t}%")
    cv = [(0, 87.8), (5, 91.9), (10, 94.0), (20, 95.5)]
    test = [(0, 92.7), (5, 96.2), (10, 97.6), (20, 97.3)]
    series(s, x, y, cv, "b")
    series(s, x, y, test, "a")
    s.text(x(10), y(97.6) - 11, "97.6%", "t small", "middle")
    s.text(x(10), y(94.0) + 21, "94.0%", "t small", "middle")
    s.text(430, 356, "Share of ROIs referred to a clinician", "s small", "middle")
    s.save("referral")


def compute():
    s = Svg(860, 372, "Hit rate against compute", "Test set, U-Net, five seeds, lcc. The cascade stops after two folds when they agree", "Hit rate against compute for four strategies")
    x, y = panel(s, 68, 72, 760, 242, (0, 225), (86, 96), [0, 50, 100, 150, 200], [86, 88, 90, 92, 94, 96], lambda t: f"{t}", lambda t: f"{t}%")
    points = [(17.6, 89.6, "R18/384 single", 10, 22), (40.9, 90.1, "R34/512 single fold", 14, 16), (105.2, 92.9, "Cascade, 2.57 folds on average", 0, -14), (204.5, 93.0, "Five-fold ensemble", -6, -14)]
    s.path("M" + "L".join(f"{x(a):.1f},{y(b):.1f}" for a, b, *_ in points), "c line", 1.25, 'stroke-dasharray="4 4"')
    for a, b, name, dx, dy in points:
        s.dot(x(a), y(b), "a", 6)
        s.text(x(a) + dx, y(b) + dy, f"{name}: {b}%", "t small", "end" if a > 200 else ("middle" if a == 105.2 else "start"))
    s.text(448, 356, "Compute per ROI (GMAC)", "s small", "middle")
    s.save("compute_tradeoff")


def learning_curve():
    s = Svg(860, 372, "The hit rate levels off at about three quarters of the data",
            "Foramen hit rate, three seeds, lcc. 75% to 100%: CV +0.1 points (p = 0.91), test +1.2 (p = 0.23)", "Learning curve")
    s.legend(24, 74, [("Test set, five-fold vote", "b"), ("Cross-validation", "a")])
    x, y = panel(s, 68, 100, 760, 214, (90, 470), (80, 95), [112, 224, 335, 447], [80, 85, 90, 95], lambda t: f"{t}", lambda t: f"{t}%")
    cv = [(112, 81.9), (224, 85.6), (335, 86.5), (447, 86.6)]
    test = [(112, 85.4), (224, 90.5), (335, 92.2), (447, 93.4)]
    series(s, x, y, test, "b")
    series(s, x, y, cv, "a")
    s.text(x(447), y(93.4) - 11, "93.4%", "t small", "middle")
    s.text(x(447), y(86.6) + 21, "86.6%", "t small", "middle")
    s.text(448, 356, "Training ROIs per fold (25%, 50%, 75%, 100% of the training groups)", "s small", "middle")
    s.save("learning_curve")


def preprocessing():
    rows = [("Hit rate (points)", (-0.18, -1.58, 1.22), (-1.46, -4.29, 1.17)),
            ("Foramen Dice (x100)", (-0.55, -1.41, 0.32), (-0.64, -3.08, 1.60)),
            ("Canal Dice (x100)", (-0.64, -1.21, -0.07), (-0.93, -1.71, -0.20))]
    height = 118 + len(rows) * 40 + 42
    s = Svg(860, height, "CLAHE did not beat the light unsharp mask",
            "CLAHE minus unsharp mask, three seeds, lcc, 95% CI. Only the canal Dice gap is significant", "Preprocessing comparison")
    s.legend(24, 74, [("Cross-validation", "a"), ("Test set, five-fold vote", "b")])
    left, right, top = 210, 830, 96
    bottom = top + len(rows) * 40
    lo, hi = -5, 2
    x = lambda v: left + (v - lo) / (hi - lo) * (right - left)
    for t in range(lo, hi + 1):
        s.line(x(t), top, x(t), bottom, "axis" if t == 0 else "grid")
        s.text(x(t), bottom + 19, f"{t:+d}" if t else "0", "s small", "middle")
    for i, (name, cv, test) in enumerate(rows):
        cy = top + i * 40 + 20
        s.text(left - 12, cy + 4, name, "t small", "end")
        for (mid, low, high), group, dy in ((cv, "a", -6), (test, "b", 7)):
            s.line(x(low), cy + dy, x(high), cy + dy, f"{group} line", 2, 'stroke-linecap="round"')
            s.dot(x(mid), cy + dy, group, 4)
    s.text((left + right) / 2, bottom + 40, "Difference, CLAHE minus unsharp mask (negative = CLAHE worse)", "s small", "middle")
    s.save("preprocessing")


def ablation():
    rows = [("Final model", 86.7, 93.0, "5 seeds"), ("Without ImageNet pretraining", 86.6, 90.8, "5 seeds"),
            ("Without strong augmentation", 84.4, 90.5, "5 seeds"), ("Plain U-Net, strong aug.", 82.3, 82.5, "seed 0"),
            ("Plain U-Net, basic aug.", 79.4, 83.2, "seed 0")]
    s = Svg(860, 118 + len(rows) * 40 + 40, "Strong augmentation and the ResNet34 encoder matter most",
            "Foramen hit rate, lcc. Pretraining leaves the hit rate unchanged but raises the Dice", "Ablation results")
    s.legend(24, 74, [("Cross-validation", "a"), ("Test set, five-fold vote", "b")])
    left, right, top = 250, 800, 96
    bottom = top + len(rows) * 40
    lo, hi = 75, 95
    x = lambda v: left + (v - lo) / (hi - lo) * (right - left)
    for t in range(lo, hi + 1, 5):
        s.line(x(t), top, x(t), bottom, "grid")
        s.text(x(t), bottom + 19, f"{t}%", "s small", "middle")
    for i, (name, cv, test, seeds) in enumerate(rows):
        cy = top + i * 40 + 20
        s.text(left - 12, cy + 4, f"{name} ({seeds})", "t small", "end")
        s.line(x(min(cv, test)), cy, x(max(cv, test)), cy, "edge", 1)
        for value, group in ((cv, "a"), (test, "b")):
            s.dot(x(value), cy, group, 5)
        s.text(x(max(cv, test)) + 9, cy + 4, f"{cv:.1f} / {test:.1f}", "s small")
    s.text((left + right) / 2, bottom + 40, "Foramen hit rate (CV / test)", "s small", "middle")
    s.save("ablation")


def roofline():
    s = Svg(760, 340, "Below the ridge point, more compute goes unused", "Schematic, not to scale; numbers measured on an RTX 3090", "Roofline schematic")
    s.path("M90 280H700", "edge", 1.25, 'marker-end="url(#arrow)"')
    s.path("M90 280V76", "edge", 1.25, 'marker-end="url(#arrow)"')
    s.text(100, 84, "Operations per second", "s small")
    s.text(700, 322, "Intensity: operations per byte moved (log scale)", "s small", "end")
    s.path("M90 260L383 110H690", "a line", 2.5)
    s.line(383, 110, 383, 280, "edge", 1, 'stroke-dasharray="4 4"')
    s.text(383, 299, "Ridge ≈ 27.6 FLOP/byte", "t small", "middle")
    s.text(550, 100, "Compute limit: about 23.9 TFLOP/s", "t small", "middle")
    s.text(210, 216, "Bandwidth limit", "t small")
    s.text(210, 232, "Slope = memory bandwidth", "s small")
    s.text(236, 268, "Memory-bound", "s small", "middle")
    s.text(540, 268, "Compute-bound", "s small", "middle")
    s.parts.append('<circle cx="333" cy="136" r="5" class="b line" fill="none" stroke-width="2"/>')
    s.dot(365, 119, "b")
    s.text(395, 160, "YOLO26n: 15.6 → 22.6 after fusion", "t small", "start")
    s.dot(451, 110, "a")
    s.text(460, 132, "U-Net: 59.9, right of the ridge", "t small")
    s.save("roofline")


def accelerator():
    s = Svg(760, 206, "Array, SRAM and DRAM bandwidth must be sized together", "", "Accelerator model")
    s.box(24, 62, 200, 86, "External DRAM", ["24.5 MB weights and features", "Sweep 4 / 12.8 / 25.6 GB/s"])
    s.box(280, 62, 200, 86, "On-chip SRAM", ["Holds the current tile", "Sweep 128 KB – 8 MB"])
    s.box(536, 62, 200, 86, "PE array", ["One INT8 MAC per PE per cycle", "Sweep 8×8 – 64×64, 500 MHz"], "a")
    for a, b in ((224, 280), (480, 536)):
        s.arrow(a, 98, b, 98)
        s.arrow(b, 114, a, 114)
    s.text(24, 182, "Layer time = max(compute time, transfer time). A larger SRAM moves the same data fewer times.", "s small")
    s.save("accelerator_model")


def hbars(s, rows, left, right, top, lo, hi, ticks, fmt, row_h=34):
    """Horizontal bars: rows = (label, value, group, note). Returns the bottom y."""
    x = lambda v: left + (v - lo) / (hi - lo) * (right - left)
    bottom = top + len(rows) * row_h
    for t in ticks:
        s.line(x(t), top, x(t), bottom, "grid")
        s.text(x(t), bottom + 19, fmt(t), "s small", "middle")
    for i, (name, value, group, note) in enumerate(rows):
        cy = top + i * row_h + row_h / 2
        s.text(left - 12, cy + 4, name, "t small", "end")
        s.parts.append(f'<rect x="{x(lo):.1f}" y="{cy - 9:.1f}" width="{x(value) - x(lo):.1f}" height="18" rx="3" class="{group} fill"/>')
        s.text(x(value) + 8, cy + 4, note, "s small")
    return bottom


def system_referral():
    # 測試集 131 側(正常牙 102、缺牙 21、FDI 錯 8),5 種子平均,原始標註;
    # 不轉介 = position_referral P0;另兩列 = deploy_pipeline 的 score_t32_640 與 score_official
    rows = [("No referral", 70.7, 27.3, 2.0), ("Numbering and side-count flags", 58.8, 9.9, 31.3), ("+ boundary-margin referral (deployed)", 54.7, 6.4, 38.9)]
    s = Svg(860, 268, "Referral trades clinician workload for fewer silent errors",
            "All 131 test sides including missing teeth and wrong numbering, five seeds, original labels", "Share of sides answered correctly, answered wrongly, or referred")
    s.legend(24, 74, [("Answered correctly", "a"), ("Answered wrongly (silent error)", "b"), ("Referred to a clinician", "c")])
    left, right, top = 290, 830, 96
    x = lambda v: left + v / 100 * (right - left)
    for t in (0, 25, 50, 75, 100):
        s.line(x(t), top, x(t), top + len(rows) * 40, "grid")
        s.text(x(t), top + len(rows) * 40 + 19, f"{t}%", "s small", "middle")
    for i, (name, ok, bad, ref) in enumerate(rows):
        cy = top + i * 40 + 20
        s.text(left - 12, cy + 4, name, "t small", "end")
        start = 0
        for value, group in ((ok, "a"), (bad, "b"), (ref, "c")):
            s.parts.append(f'<rect x="{x(start):.1f}" y="{cy - 11:.1f}" width="{x(start + value) - x(start):.1f}" height="22" class="{group} fill"/>')
            if value >= 6:
                s.parts.append(f'<text x="{x(start + value / 2):.1f}" y="{cy + 4:.1f}" text-anchor="middle" style="fill:#fff;font-size:12px">{value:.1f}%</text>')
            start += value
    s.text((left + right) / 2, top + len(rows) * 40 + 40, "Share of all test sides", "s small", "middle")
    s.save("system_referral")


def low_bit():
    # CV 命中差(點)與 95% CI;判準 = 不掉超過約 1 點(事先定)。EXPERIMENTS.md「硬體: 結構化稀疏與 4-bit」
    rows = [("W4A8 QAT vs INT8 (seed 0)", (0.00, -1.27, 1.30)), ("W4A4 QAT vs INT8 (seed 0)", (-6.08, -8.95, -3.30)),
            ("2:4 sparse FP32 vs FP32 (seed 0)", (0.54, -0.56, 1.73)), ("2:4 + INT8 vs dense INT8 (5 seeds)", (0.50, -0.15, 1.16)),
            ("2:4 + W4A8 vs dense INT8 (seed 0)", (-1.07, -2.72, 0.55)), ("2:4 fixed point vs integer (5 seeds)", (0.04, -0.19, 0.27))]
    height = 118 + len(rows) * 36 + 44
    s = Svg(860, height, "INT8 with 2:4 sparsity keeps the hit rate; 4-bit activations do not",
            "Cross-validation hit rate difference, 95% CI. Dashed: the pre-set margin of about −1 point", "Hit rate change for low-bit and sparse U-Nets")
    left, right, top = 300, 830, 80
    bottom = top + len(rows) * 36
    lo, hi = -10, 3
    x = lambda v: left + (v - lo) / (hi - lo) * (right - left)
    for t in range(lo, hi + 1, 2):
        s.line(x(t), top, x(t), bottom, "grid")
        s.text(x(t), bottom + 19, f"{t:+d}" if t else "0", "s small", "middle")
    s.line(x(0), top, x(0), bottom, "axis")
    s.line(x(-1), top, x(-1), bottom, "edge", 1, 'stroke-dasharray="4 4"')
    for i, (name, (mid, low, high)) in enumerate(rows):
        cy = top + i * 36 + 18
        group = "b" if high < -1 or mid < -1 else "a"
        s.text(left - 12, cy + 4, name, "t small", "end")
        s.line(x(low), cy, x(high), cy, f"{group} line", 2, 'stroke-linecap="round"')
        s.dot(x(mid), cy, group, 5)
    s.text((left + right) / 2, bottom + 40, "Hit rate difference (points); 2:4 keeps the stem and the 32→3 output layer dense", "s small", "middle")
    s.save("low_bit")


def deploy_latency():
    # 筆電 i5-13500H + RTX 4050,測試集 103 張;同一時段內才比(time9: 11/16 09:50–10:31,關掉 Windows 背景節流;time8: 11/15 10:41–10:49)
    s = Svg(860, 470, "Each speed-up in the deployed pipeline, measured in one session",
            "Laptop, mean wall-clock time per radiograph, 103 test radiographs, three rounds, Windows power throttling disabled for the process", "Latency per radiograph by configuration")
    rows1 = [("Integrated pipeline (13 Nov)", 1216, "c", "1216 ms"), ("+ 32-class model INT8, union", 1063, "c", "1063 ms"),
             ("+ box-only mask upsampling", 940, "c", "940 ms"), ("+ 32-class model at 640 px", 890, "c", "890 ms"),
             ("+ fast argmax (deployed)", 802, "a", "802 ms"), ("Deployed, U-Net FP32 instead of INT8", 1879, "b", "1879 ms (INT8: 2.34× faster)")]
    s.text(24, 84, "Session 1: CPU (OpenVINO)", "t bold")
    b = hbars(s, rows1, 300, 700, 94, 0, 2400, [0, 500, 1000, 1500, 2000], lambda t: f"{t}", 32)
    rows2 = [("CPU, deployed (INT8)", 800, "a", "800 ms"), ("RTX 4050, U-Net FP32", 217, "b", "217 ms"), ("RTX 4050, U-Net FP16", 194, "b", "194 ms")]
    s.text(24, b + 50, "Session 2: CPU and laptop GPU", "t bold")
    b2 = hbars(s, rows2, 300, 700, b + 60, 0, 2400, [0, 500, 1000, 1500, 2000], lambda t: f"{t}", 32)
    s.text(530, b2 + 40, "Milliseconds per radiograph", "s small", "middle")
    s.save("deploy_latency")


def cpu_throttling():
    # throttle_long.ps1(11/16): 新開的最小化背景視窗,每 4 分鐘一次 15 秒基準,交替 正常 / 關掉 power throttling
    normal = [(0.1, 223.2), (8.1, 321.9), (16.1, 336.8), (24.1, 321.7), (32.1, 322.3)]
    opt_out = [(4.1, 194.1), (12.1, 189.4), (20.1, 226.6), (28.1, 234.0)]
    s = Svg(860, 372, "A background window slows the laptop down after a few minutes",
            "i5-13500H, stage-1 model FP32, 15-second benchmark every 4 minutes in a minimized window", "Stage-1 latency against time spent in a background window")
    s.legend(24, 74, [("Default", "b"), ("Power throttling disabled for the process", "a")])
    x, y = panel(s, 68, 100, 760, 214, (-1, 34), (150, 360), [0, 8, 16, 24, 32], [150, 200, 250, 300, 350], lambda v: f"{v} min", lambda v: f"{v} ms")
    series(s, x, y, normal, "b")
    series(s, x, y, opt_out, "a")
    s.text(x(9), y(285), "Windows switches the process to efficiency scheduling (EcoQoS)", "t small", "start")
    s.text(x(33), y(165), "The 4-thread U-Net stays at 46–52 ms in both", "s small", "end")
    s.text(448, 356, "Minutes in a background window", "s small", "middle")
    s.save("cpu_throttling")

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for make in (pipeline, hit_threshold, referral, compute, learning_curve, preprocessing, ablation, roofline, accelerator,
                 system_referral, low_bit, deploy_latency, cpu_throttling):
        make()
    print(sorted(p.name for p in OUT.glob("*.svg")))
