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
    s.box(24, 182, 190, 74, "Five-fold vote", ["Pixel-wise majority", "stage2/kfold_pano.py"])
    s.box(24, 290, 190, 74, "Largest region (lcc)", ["One foramen per ROI", "stage2/mf_postproc.py"])
    s.box(254, 290, 270, 74, "Foramen and tooth position", ["Centroid, canal mask, FDI tooth position", "position/mf_position.py"], "b")
    s.box(564, 290, 272, 74, "Clinician review", ["Model disagreement flags uncertain cases", "analysis/pano_uncertainty.py"])
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
    s = Svg(860, 372, "More labels would add little", "Foramen Dice by training size, five-fold mean, seed 0. Earlier patient-level split", "Learning curve")
    s.legend(24, 74, [("Cross-validation", "a"), ("Test set", "b")])
    x, y = panel(s, 68, 100, 760, 214, (90, 470), (.55, .75), [113, 225, 336, 447], [.55, .6, .65, .7, .75], lambda t: f"{t}", lambda t: f"{t:.2f}")
    cv = [(113, .651), (225, .676), (336, .699), (447, .701)]
    test = [(113, .591), (225, .638), (336, .666), (447, .670)]
    series(s, x, y, test, "b")
    series(s, x, y, cv, "a")
    s.text(x(447), y(.701) - 11, "0.701", "t small", "middle")
    s.text(x(447), y(.670) + 21, "0.670", "t small", "middle")
    s.text(448, 356, "Training ROIs", "s small", "middle")
    s.save("learning_curve")


def preprocessing():
    rows = [("CLAHE", (.0115, -.0023, .0262), (.0104, -.0121, .0353)), ("Gamma", (.0073, -.0075, .0218), (-.0080, -.0312, .0133)),
            ("Gaussian", (.0057, -.0067, .0183), (-.0026, -.0232, .0196)), ("Unsharp mask", (.0006, -.0137, .0148), (-.0022, -.0237, .0177)),
            ("Median", (.0003, -.0126, .0125), (-.0039, -.0215, .0157)), ("Bilateral", (.0000, -.0134, .0134), (-.0078, -.0305, .0170)),
            ("BF + gamma + CLAHE + UM", (-.0131, -.0275, .0011), (-.0224, -.0569, .0109)), ("Laplacian", (-.0142, -.0285, -.0004), (-.0051, -.0215, .0117)),
            ("Histogram equalization", (-.0203, -.0369, -.0053), (-.0251, -.0461, -.0032))]
    height = 118 + len(rows) * 32 + 42
    s = Svg(860, height, "No preprocessing method beat the original image", "Foramen Dice difference from the original image, 95% CI, seed 0. Earlier patient-level split. Grey band: ±0.02", "Preprocessing forest plot")
    s.legend(24, 74, [("Cross-validation", "a"), ("Test set", "b")])
    left, right, top = 210, 830, 96
    bottom = top + len(rows) * 32
    x = lambda v: left + (v + .06) / .1 * (right - left)
    s.parts.append(f'<rect x="{x(-.02):.1f}" y="{top}" width="{x(.02) - x(-.02):.1f}" height="{bottom - top}" class="c tint"/>')
    for t in (-.06, -.04, -.02, 0, .02, .04):
        s.line(x(t), top, x(t), bottom, "axis" if t == 0 else "grid")
        s.text(x(t), bottom + 19, f"{t:+.2f}" if t else "0", "s small", "middle")
    for i, (name, cv, test) in enumerate(rows):
        cy = top + i * 32 + 16
        s.text(left - 12, cy + 4, name, "t small", "end")
        for (mid, low, high), group, dy in ((cv, "a", -5), (test, "b", 6)):
            s.line(x(low), cy + dy, x(high), cy + dy, f"{group} line", 2, 'stroke-linecap="round"')
            s.dot(x(mid), cy + dy, group, 4)
    s.text((left + right) / 2, bottom + 40, "Foramen Dice difference from the original image", "s small", "middle")
    s.save("preprocessing")


def ablation():
    rows = [("Final model", .701, .689), ("Without pretraining", .694, .690), ("Without strong augmentation", .675, .680), ("Plain U-Net, strong aug.", .628, .542), ("Plain U-Net, basic aug.", .551, .469)]
    s = Svg(860, 118 + len(rows) * 46 + 34, "The ResNet34 encoder and strong augmentation matter most", "Foramen Dice, seed 0. Earlier patient-level split", "Ablation results")
    s.legend(24, 74, [("Cross-validation", "a"), ("Test set, five-fold vote", "b")])
    left, right, top = 210, 800, 96
    bottom = top + len(rows) * 46
    x = lambda v: left + v / .8 * (right - left)
    for t in (0, .2, .4, .6, .8):
        s.line(x(t), top, x(t), bottom, "axis" if t == 0 else "grid")
        s.text(x(t), bottom + 19, f"{t:.1f}", "s small", "middle")
    for i, (name, cv, test) in enumerate(rows):
        y0 = top + i * 46 + 7
        s.text(left - 12, y0 + 19, name, "t small", "end")
        for j, (value, group) in enumerate(((cv, "a"), (test, "b"))):
            y1 = y0 + j * 16
            width = x(value) - left
            s.parts.append(f'<path d="M{left},{y1}h{width - 4:.1f}a4,4 0 0 1 4,4v6a4,4 0 0 1 -4,4h{-(width - 4):.1f}z" class="{group} fill"/>')
            s.text(x(value) + 6, y1 + 11, f"{value:.3f}", "s small")
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


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for make in (pipeline, hit_threshold, referral, compute, learning_curve, preprocessing, ablation, roofline, accelerator):
        make()
    print(sorted(p.name for p in OUT.glob("*.svg")))
