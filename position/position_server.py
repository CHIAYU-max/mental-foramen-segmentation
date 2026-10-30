import argparse
import csv
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 頦孔牙位(Tebo & Telford 六分類)的瀏覽器標註工具。
# 只用 Python 標準函式庫: 可以在伺服器上跑,也可以把本檔與 mf_position/ 資料夾(items.json、images/)複製到其他電腦執行。
# 資料由 `python -m position.mf_position prepare` 產生;每位標註者一個檔案 mf_position/labels_<名字>.csv,每次判定即時寫回。
#   python -m position.position_server --rater <名字>   再用瀏覽器開 http://localhost:8766
# 預設只顯示頦孔 GT 輪廓;牙齒輪廓、FDI 編號與參考線來自模型,為避免標註偏向規則預設隱藏(A 切換),
#   標註當下是否開著會記在 axes_shown 欄位。
# 2026-10-25: 按 E 標記「FDI 編號有誤」(fdi_wrong 欄位,與牙位分開存),用來評估 FDI 編號(例如 fdi_numbering.py 的空隙判斷)。
#   建議流程: 先不看疊圖標牙位,再按 A 打開疊圖,編號錯就按 E。
# 2026-10-27: 參考線改顯示方法 A(牙根長軸往下延伸,items.json 的 lines_a)。2026-10-28: x6 改用整顆牙的長軸(使用者決定)。
#   缺牙(方法 A 缺 x4、x5、x6 任一條參考線,規則無法判定;614 個中 19 個)先略過,不放進標註順序;
#   模型沒抓到的缺牙,標註者看到時按 M 略過。
PORT = 8766
FIELDS = ["alias", "position", "axes_shown", "fdi_wrong", "time"]
CODES = list("123456MX")

PAGE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>頦孔牙位標註</title>
<style>
body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; padding: 8px 16px; background: #1c1c1c; }
header b { font-size: 18px; }
.bar { flex: 1; min-width: 160px; height: 6px; background: #333; border-radius: 3px; overflow: hidden; }
.bar i { display: block; height: 100%; background: #4a9eff; }
.state { padding: 2px 10px; border-radius: 10px; background: #333; }
.state.set { background: #1f5c1f; }
.wrap { display: flex; gap: 12px; padding: 10px; }
main { flex: 1; display: flex; justify-content: center; align-items: flex-start; min-width: 0; }
canvas { max-width: 100%; max-height: calc(100vh - 70px); }
aside { width: 280px; flex: none; display: flex; flex-direction: column; gap: 6px; }
button { font-size: 15px; padding: 8px 10px; border-radius: 6px; border: 1px solid #555; background: #2a2a2a; color: #eee; cursor: pointer; text-align: left; }
button.on { border-color: #4a9eff; background: #18324f; }
#choices button { display: block; width: 100%; margin-bottom: 6px; }
.toggles { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px; }
.toggles button { font-size: 13px; text-align: center; }
small { color: #aaa; line-height: 1.5; }
</style></head><body>
<header><b id="alias"></b><span id="side"></span><span id="pos"></span><span class="state" id="state"></span><div class="bar"><i id="fill"></i></div><span id="done"></span></header>
<div class="wrap"><main><canvas id="cv"></canvas></main>
<aside id="side-panel">
<div id="choices"></div>
<div class="toggles">
<button id="t-axes">牙齒 / 參考線 (A)</button><button id="t-mf">頦孔輪廓 (G)</button>
<button id="t-contrast">提高對比 (C)</button><button id="t-full">整張全景片 (F)</button>
<button id="back">← 上一個 (B)</button><button id="next">下一個 →</button>
<button id="t-fdi" style="grid-column: span 2">FDI 編號有誤 (E)</button>
</div>
<small>判定(2026-10-28 使用者決定): 沿牙根的<b>長軸(中線)往下延伸</b>畫一條線(x6 用整顆牙)。<br>
線通過頦孔<b>中央那一半</b> → 該牙下方;只擦到邊緣、相切或沒碰到 → 依頦孔中心落在哪兩條線之間,判為兩牙之間。<br>
參考線由模型算出,只供參考,請以影像判斷。模型判定缺牙的已自動略過;若仍看到缺牙,按 M 略過。<br>
標完牙位後可按 A 打開疊圖,若綠色的 FDI 編號與實際牙齒不符,按 E 標記(再按一次取消)。</small>
</aside></div>
<script>
const data = __DATA__, labels = __LABELS__, fdiWrong = __FDI__, order = data.order;
const NAMES = q => ({"1": `${q}4 前方`, "2": `${q}4 下方`, "3": `${q}4–${q}5 之間`, "4": `${q}5 下方`,
  "5": `${q}5–${q}6 之間`, "6": `${q}6 下方`, "M": "缺牙,略過", "X": "影像不清,無法判定"});
let i = order.findIndex(a => !(a in labels)); if (i < 0) i = order.length - 1;
const view = { axes: false, mf: true, contrast: false, full: false };
const $ = id => document.getElementById(id), cv = $("cv"), ctx = cv.getContext("2d");
const cache = {};
function load(src) { if (!cache[src]) { cache[src] = new Image(); cache[src].src = "/img/" + src; } return cache[src]; }
function poly(pts, color, width, close = true) {
  ctx.beginPath(); pts.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  if (close) ctx.closePath(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
}
function draw() {
  const it = data.items[order[i]], img = load(view.full ? it.full : it.image);
  if (!img.complete) { img.onload = draw; return; }
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  ctx.filter = view.contrast ? "contrast(1.8) brightness(1.1)" : "none";
  ctx.drawImage(img, 0, 0); ctx.filter = "none";
  if (view.full) { const [x, y, w, h] = it.crop; ctx.strokeStyle = "#4a9eff"; ctx.lineWidth = 3; ctx.strokeRect(x, y, w, h); return; }
  const lw = Math.max(1.5, cv.width / 600);
  if (view.axes) {
    ctx.font = `${Math.round(lw * 9)}px sans-serif`;
    it.teeth.forEach(t => { poly(t.poly, "rgba(80,220,120,.8)", lw);
      const xs = t.poly.map(p => p[0]), ys = t.poly.map(p => p[1]);
      ctx.fillStyle = "#5f5"; ctx.fillText(t.fdi, (Math.min(...xs) + Math.max(...xs)) / 2 - lw * 6, Math.min(...ys) - lw * 3); });
    it.lines_a.forEach(l => { ctx.setLineDash([lw * 6, lw * 4]); poly(l.p, "#3cf", lw, false); ctx.setLineDash([]);
      ctx.fillStyle = "#3cf"; const [[x0, y0], [x1, y1]] = l.p, x = x0 + 0.85 * (x1 - x0), y = y0 + 0.85 * (y1 - y0);
      ctx.fillText(l.name, Math.min(x + lw * 3, cv.width - ctx.measureText(l.name).width - 2), y); });
  }
  if (view.mf) poly(it.mf, "#ffd23f", lw);
}
function show() {
  const a = order[i], it = data.items[a], names = NAMES(it.quadrant);
  $("alias").textContent = a;
  $("side").textContent = it.quadrant === 4 ? "病人右側(影像左側)" : "病人左側(影像右側)";
  $("pos").textContent = (i + 1) + " / " + order.length;
  const v = labels[a];
  $("state").textContent = (v ? v + " " + names[v] : "未標註") + (fdiWrong[a] ? "|FDI 有誤" : "");
  $("t-fdi").className = fdiWrong[a] ? "on" : "";
  $("state").className = "state" + (v ? " set" : "");
  const n = Object.keys(labels).length;
  $("done").textContent = "已標 " + n;
  $("fill").style.width = (100 * n / order.length) + "%";
  $("choices").innerHTML = "";
  for (const c of __CODES__) {
    const b = document.createElement("button");
    b.textContent = `${c}  ${names[c]}`; b.className = v === c ? "on" : ""; b.onclick = () => mark(c);
    $("choices").appendChild(b);
  }
  for (const k in view) $("t-" + k).className = view[k] ? "on" : "";
  draw();
  if (i + 1 < order.length) load(data.items[order[i + 1]].image);
}
async function mark(c) {
  const a = order[i];
  const r = await fetch("/label", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ alias: a, position: c, axes_shown: view.axes ? 1 : 0 }) });
  if (!r.ok) { alert("存檔失敗,請確認伺服器程式仍在執行"); return; }
  labels[a] = c;
  if (i < order.length - 1) i++;
  show();
}
async function flagFdi() {
  const a = order[i], value = fdiWrong[a] ? 0 : 1;
  const r = await fetch("/fdi", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ alias: a, fdi_wrong: value }) });
  if (!r.ok) { alert("存檔失敗,請確認伺服器程式仍在執行"); return; }
  if (value) fdiWrong[a] = 1; else delete fdiWrong[a];
  show();
}
$("t-fdi").onclick = flagFdi;
function toggle(k) { view[k] = !view[k]; show(); }
$("t-axes").onclick = () => toggle("axes"); $("t-mf").onclick = () => toggle("mf");
$("t-contrast").onclick = () => toggle("contrast"); $("t-full").onclick = () => toggle("full");
$("back").onclick = () => { if (i > 0) { i--; show(); } };
$("next").onclick = () => { if (i < order.length - 1) { i++; show(); } };
document.addEventListener("keydown", e => {
  const k = e.key.toUpperCase();
  if (__CODES__.includes(k)) mark(k);
  else if (k === "A") toggle("axes"); else if (k === "G") toggle("mf");
  else if (k === "C") toggle("contrast"); else if (k === "F") toggle("full"); else if (k === "E") flagFdi();
  else if (k === "B" || e.key === "ArrowLeft") $("back").click();
  else if (e.key === "ArrowRight") $("next").click();
});
show();
</script></body></html>"""


def load_labels(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["alias"]: {**r, "fdi_wrong": r.get("fdi_wrong") or 0} for r in csv.DictReader(f) if r["position"] or r.get("fdi_wrong") == "1"}


def save_labels(path, order, labels):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(labels[a] for a in order if a in labels)


def has_reference_lines(item):
    # 方法 A 的 x4、x5、x6 參考線都在(線名如 "34"、"36",第 2 個字是牙號)才標;缺任一顆 = 缺牙,略過
    return {seg["name"][1] for seg in item["lines_a"]} >= set("456")


class Handler(BaseHTTPRequestHandler):
    data, labels, images, labels_path = {}, {}, None, None

    def reply(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            page = (PAGE.replace("__DATA__", json.dumps(self.data, ensure_ascii=False))
                    .replace("__LABELS__", json.dumps({a: r["position"] for a, r in self.labels.items() if r["position"]}))
                    .replace("__FDI__", json.dumps({a: 1 for a, r in self.labels.items() if str(r["fdi_wrong"]) == "1"}))
                    .replace("__CODES__", json.dumps(CODES)))
            self.reply(200, page.encode(), "text/html; charset=utf-8")
            return
        name = self.path[5:] if self.path.startswith("/img/") else ""
        path = self.images / name
        if name and "/" not in name and ".." not in name and name.endswith(".jpg") and path.exists():
            self.reply(200, path.read_bytes(), "image/jpeg")
        else:
            self.reply(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/label", "/fdi"):
            self.reply(404, b"not found", "text/plain")
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        alias = body.get("alias")
        if alias not in self.data["items"] or (self.path == "/label" and body.get("position") not in CODES):
            self.reply(400, b"bad request", "text/plain")
            return
        row = self.labels.get(alias, {"alias": alias, "position": "", "axes_shown": 0, "fdi_wrong": 0})
        if self.path == "/label":
            row.update(position=body["position"], axes_shown=int(bool(body.get("axes_shown"))))
        else:
            row["fdi_wrong"] = int(bool(body.get("fdi_wrong")))
        row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.labels[alias] = row
        save_labels(self.labels_path, self.data["order"], self.labels)
        self.reply(200, b"ok", "text/plain")

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rater", required=True, help="標註者名稱(英數字),結果存在 labels_<rater>.csv")
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parents[1] / "mf_position"))
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if not args.rater.replace("_", "").isalnum():
        raise SystemExit("--rater 只能用英數字與底線")
    root = Path(args.dir)
    if not (root / "items.json").exists():
        raise SystemExit(f"找不到 {root / 'items.json'},請先執行 python -m position.mf_position prepare")
    Handler.data = json.loads((root / "items.json").read_text(encoding="utf-8"))
    order = Handler.data["order"]
    Handler.data["order"] = [a for a in order if has_reference_lines(Handler.data["items"][a])]
    skipped = len(order) - len(Handler.data["order"])
    Handler.images = root / "images"
    Handler.labels_path = root / f"labels_{args.rater}.csv"
    Handler.labels = load_labels(Handler.labels_path)
    total = len(Handler.data["order"])
    # 只接受本機連線;從自己的電腦連線請透過 VS Code 的連接埠轉送或 SSH 通道
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"{args.rater}: 已標 {len(Handler.labels)} / {total} 個(缺牙略過 {skipped} 個)")
    print(f"請用瀏覽器開啟 http://localhost:{args.port}  (按 Ctrl+C 結束,結果已即時存檔)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"\n已標 {len(Handler.labels)} / {total} 個,結果存在 {Handler.labels_path}")


if __name__ == "__main__":
    main()
