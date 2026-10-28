import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from visibility_grading import GRADES_CSV, IMAGES

# 頦孔可辨識性的瀏覽器判定工具(伺服器沒有圖形介面時使用)。
# 在伺服器上執行後,用瀏覽器開 http://localhost:<port>,每次判定都即時寫回 grades.csv。
#   1 = 看得出來    0 = 看不出來    B / ← = 上一張    → = 下一張(不判定)
PORT = 8765
TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp"}

PAGE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>頦孔可辨識性判定</title>
<style>
body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: center; padding: 10px 16px; background: #1c1c1c; }
header b { font-size: 18px; }
.bar { flex: 1; min-width: 160px; height: 6px; background: #333; border-radius: 3px; overflow: hidden; }
.bar i { display: block; height: 100%; background: #4a9eff; }
main { display: flex; justify-content: center; padding: 12px; }
img { max-width: 100%; max-height: calc(100vh - 150px); image-rendering: auto; }
footer { display: flex; justify-content: center; gap: 12px; padding: 8px; }
button { font-size: 16px; padding: 10px 22px; border-radius: 6px; border: 1px solid #555; background: #2a2a2a; color: #eee; cursor: pointer; }
button.yes { border-color: #3c8f3c; } button.no { border-color: #a04040; }
.state { padding: 2px 10px; border-radius: 10px; background: #333; }
.state.v1 { background: #1f5c1f; } .state.v0 { background: #6b2323; }
</style></head><body>
<header><b id="alias"></b><span id="pos"></span><span class="state" id="state"></span><div class="bar"><i id="fill"></i></div><span id="done"></span></header>
<main><img id="img" alt=""></main>
<footer><button id="back">← 上一張 (B)</button><button class="yes" id="yes">看得出來 (1)</button><button class="no" id="no">看不出來 (0)</button><button id="next">下一張 →</button></footer>
<script>
const order = __ORDER__, grades = __GRADES__;
let i = order.findIndex(a => !(a in grades)); if (i < 0) i = order.length - 1;
const $ = id => document.getElementById(id);
function show() {
  const a = order[i];
  $("img").src = "/img/" + a;
  $("alias").textContent = a;
  $("pos").textContent = (i + 1) + " / " + order.length;
  const g = grades[a];
  $("state").textContent = g === "1" ? "看得出來" : g === "0" ? "看不出來" : "未判定";
  $("state").className = "state" + (g ? " v" + g : "");
  const n = Object.keys(grades).length;
  $("done").textContent = "已判定 " + n;
  $("fill").style.width = (100 * n / order.length) + "%";
  if (i + 1 < order.length) new Image().src = "/img/" + order[i + 1];
}
async function grade(v) {
  const a = order[i];
  const r = await fetch("/grade", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ alias: a, value: v }) });
  if (!r.ok) { alert("存檔失敗,請確認伺服器程式仍在執行"); return; }
  grades[a] = v;
  if (i < order.length - 1) i++;
  show();
}
$("yes").onclick = () => grade("1");
$("no").onclick = () => grade("0");
$("back").onclick = () => { if (i > 0) { i--; show(); } };
$("next").onclick = () => { if (i < order.length - 1) { i++; show(); } };
document.addEventListener("keydown", e => {
  if (e.key === "1") grade("1");
  else if (e.key === "0") grade("0");
  else if (e.key === "b" || e.key === "B" || e.key === "ArrowLeft") $("back").click();
  else if (e.key === "ArrowRight") $("next").click();
});
show();
</script></body></html>"""


def load():
    with open(GRADES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    order = [r["alias"] for r in rows]
    grades = {r["alias"]: r["mf_visible"] for r in rows if r["mf_visible"].strip() in ("0", "1")}
    return order, grades


def save(order, grades):
    with open(GRADES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "mf_visible"])
        for alias in order:
            writer.writerow([alias, grades.get(alias, "")])


class Handler(BaseHTTPRequestHandler):
    order, grades, paths = [], {}, {}

    def reply(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            page = PAGE.replace("__ORDER__", json.dumps(self.order)).replace("__GRADES__", json.dumps(self.grades))
            self.reply(200, page.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/img/") and self.path[5:] in self.paths:
            path = self.paths[self.path[5:]]
            self.reply(200, path.read_bytes(), TYPES[path.suffix.lower()])
        else:
            self.reply(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/grade":
            self.reply(404, b"not found", "text/plain")
            return
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if data.get("alias") not in self.paths or data.get("value") not in ("0", "1"):
            self.reply(400, b"bad request", "text/plain")
            return
        self.grades[data["alias"]] = data["value"]
        save(self.order, self.grades)
        self.reply(200, b"ok", "text/plain")

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if not GRADES_CSV.exists():
        raise SystemExit(f"找不到 {GRADES_CSV},請先執行 python visibility_grading.py prepare")
    Handler.order, Handler.grades = load()
    Handler.paths = {p.stem: p for p in IMAGES.iterdir() if p.suffix.lower() in TYPES}
    # 只接受本機連線;從自己的電腦連線請透過 VS Code 的連接埠轉送或 SSH 通道
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"已判定 {len(Handler.grades)} / {len(Handler.order)} 張")
    print(f"請用瀏覽器開啟 http://localhost:{args.port}  (按 Ctrl+C 結束,判定結果已即時存檔)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"\n已判定 {len(Handler.grades)} / {len(Handler.order)} 張,結果存在 {GRADES_CSV}")


if __name__ == "__main__":
    main()
