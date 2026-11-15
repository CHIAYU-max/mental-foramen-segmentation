"""2026-11-14 牙位定義的疊圖判讀工具(使用者要求;只用標準函式庫)

11/5 產生的 30 張疊圖(compare_results/mf_position/midline/overlays/NN.jpg: 現行「中央一半」與「中線規則」各自的判定)原本要手動填 overlays.csv,
這裡改成瀏覽器一鍵判讀: C = 現行規則對、M = 中線規則對、N = 兩個都不對;結果即時寫回 overlays.csv 的 your_call 欄。病人影像只在伺服器。
  python -m position.midline_review_server   再用瀏覽器開 http://localhost:8768(筆電用「牙位定義判讀」捷徑,會開 SSH 通道)
判讀完執行 python midline_review_report.py 看統計與事先定的決定規則。
"""
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "compare_results" / "mf_position" / "midline" / "overlays"
CSV = DIR / "overlays.csv"
CALL = "your_call(current/midline/neither)"
PORT = 8768
CODES = {"C": "current", "M": "midline", "N": "neither"}

PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>牙位定義判讀</title>
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:8px 16px;background:#1c1c1c}
.wrap{display:flex;gap:12px;padding:10px}main{flex:1;text-align:center}img{max-width:100%;max-height:calc(100vh - 70px)}
aside{width:280px;flex:none;display:flex;flex-direction:column;gap:6px}
button{font-size:15px;padding:8px 10px;border-radius:6px;border:1px solid #555;background:#2a2a2a;color:#eee;cursor:pointer;text-align:left}
button.on{border-color:#4a9eff;background:#18324f}small{color:#aaa;line-height:1.5}</style></head><body>
<header><b id="no"></b><span id="pos"></span><span id="state"></span><span id="done"></span></header>
<div class="wrap"><main><img id="im"></main><aside>
<button id="bC">C  現行規則(中央一半)的答案對</button><button id="bM">M  中線規則的答案對</button><button id="bN">N  兩個都不對 / 無法判斷</button>
<button id="back">← 上一張 (B)</button><button id="next">下一張 →</button>
<small>圖上: 青色 = 第一階段的參考線(牙根長軸延伸);黃色虛線 = 兩條參考線的中線;白色 = 牙齒輪廓;洋紅 = 根尖寬度往下延伸;紅色 = 頦孔。<br>
上方文字列出兩種規則各自判的牙位。請依影像判斷哪一個答案比較符合臨床上「頦孔在哪顆牙下方 / 哪兩顆牙之間」。<br>
共 30 張,其中 24 張兩種規則的答案不同。</small></aside></div>
<script>
const rows = __ROWS__; let i = rows.findIndex(r => !r.call); if (i < 0) i = rows.length - 1;
const $ = id => document.getElementById(id);
function show() { const r = rows[i]; $("no").textContent = r.no; $("pos").textContent = (i + 1) + " / " + rows.length;
  $("state").textContent = r.call ? "已判: " + r.call : "未判"; $("done").textContent = "已判 " + rows.filter(x => x.call).length;
  $("im").src = "/img/" + r.no + ".jpg"; for (const k of "CMN") $("b" + k).className = r.call === __CODES__[k] ? "on" : ""; }
async function mark(k) { const r = rows[i];
  const res = await fetch("/call", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({no: r.no, call: __CODES__[k]})});
  if (!res.ok) { alert("存檔失敗"); return; } r.call = __CODES__[k]; if (i < rows.length - 1) i++; show(); }
for (const k of "CMN") $("b" + k).onclick = () => mark(k);
$("back").onclick = () => { if (i > 0) { i--; show(); } }; $("next").onclick = () => { if (i < rows.length - 1) { i++; show(); } };
document.addEventListener("keydown", e => { const k = e.key.toUpperCase(); if ("CMN".includes(k) && k.length === 1) mark(k);
  else if (k === "B" || e.key === "ArrowLeft") $("back").click(); else if (e.key === "ArrowRight") $("next").click(); });
show();
</script></body></html>"""


def load():
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(rows):
    with open(CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            rows = [{"no": r["no"], "call": r[CALL]} for r in load()]
            page = PAGE.replace("__ROWS__", json.dumps(rows)).replace("__CODES__", json.dumps(CODES))
            self.reply(200, page.encode(), "text/html; charset=utf-8")
            return
        name = self.path[5:] if self.path.startswith("/img/") else ""
        path = DIR / name
        if name and "/" not in name and ".." not in name and name.endswith(".jpg") and path.exists():
            self.reply(200, path.read_bytes(), "image/jpeg")
        else:
            self.reply(404, b"not found", "text/plain")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        rows = load()
        hit = [r for r in rows if r["no"] == body.get("no")]
        if self.path != "/call" or not hit or body.get("call") not in CODES.values():
            self.reply(400, b"bad request", "text/plain")
            return
        hit[0][CALL] = body["call"]
        save(rows)
        self.reply(200, b"ok", "text/plain")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"已判 {sum(bool(r[CALL]) for r in load())} / {len(load())};http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
