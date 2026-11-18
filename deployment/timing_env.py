"""2026-11-17 計時環境記錄(使用者要求): 可用記憶體、分頁檔用量、Windows 電源模式、OMEN Gaming Hub 模式。

背景: 筆電(HP Victus 15-fa1xxx)的計時要固定在同一組軟體設定下比較(EXPERIMENTS.md「計時環境: NUMA、swap、BIOS、虛擬化」)。
  - 分頁檔: 正常時只有一百多 MB;計時前後明顯變大代表當時記憶體不夠,那一輪不能用。
  - OMEN 模式(Eco / 平衡 / 效能)會改 CPU 功耗上限,程式讀不到,要手動指定: 環境變數 OMEN_MODE,
    或 deploy_pipeline.py --omen-mode(會設定 OMEN_MODE)。沒指定時記為 unknown。
用法:
  snapshot() → dict(計時前後各呼叫一次)
  begin()    → 記下開始時的 snapshot,程式結束時把「開始 / 結束」寫一列到 compare_results/deploy_eval/timing_env_log.csv。
               no_throttle.disable() 會自動呼叫,所以所有計時程式都有記錄;設 TIMING_ENV_LOG=0 可關。
只記錄,不影響計時結果。
"""
import atexit
import csv
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "compare_results" / "deploy_eval" / "timing_env_log.csv"
OVERLAYS = {"ded574b5-45a0-4f42-8737-46345c09c238": "最佳效能", "961cc777-2547-4f9d-8174-7d86181b8a7a": "最佳省電",
            "00000000-0000-0000-0000-000000000000": "平衡", "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "更好的效能"}
OMEN_MODES = ("eco", "balanced", "performance", "unknown")


def omen_mode():
    m = os.environ.get("OMEN_MODE", "unknown").strip().lower()
    return m if m in OMEN_MODES else f"invalid:{m}"


def _windows(env):
    import ctypes
    import winreg

    class Status(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte), ("system", ctypes.c_ubyte),
                    ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    st = Status()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st))
    env.update(power={0: "battery", 1: "AC"}.get(st.ac, "unknown"), battery=st.percent, charging=int(bool(st.flag & 8)))
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes")
        guid = winreg.QueryValueEx(key, "ActiveOverlayAcPowerScheme" if st.ac == 1 else "ActiveOverlayDcPowerScheme")[0]
        env["mode"] = OVERLAYS.get(guid.lower(), guid)
    except OSError:
        pass

    class Mem(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong), ("total_commit", ctypes.c_ulonglong), ("avail_commit", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong), ("avail_ext", ctypes.c_ulonglong)]
    m = Mem()
    m.length = ctypes.sizeof(Mem)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        env.update(mem_avail_gb=round(m.avail_phys / 2**30, 2), mem_total_gb=round(m.total_phys / 2**30, 2), mem_load_pct=m.load)
    try:  # 分頁檔目前用量(MB);約 1 秒,只在計時前後呼叫
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_PageFileUsage | Measure-Object CurrentUsage -Sum).Sum"],
                           capture_output=True, text=True, timeout=20)
        env["pagefile_mb"] = int(r.stdout.strip())
    except Exception:
        pass


def _linux(env):
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        kb = lambda k: int(info[k].split()[0])  # noqa: E731
        env.update(mem_avail_gb=round(kb("MemAvailable") / 2**20, 2), mem_total_gb=round(kb("MemTotal") / 2**20, 2),
                   pagefile_mb=round((kb("SwapTotal") - kb("SwapFree")) / 1024))
    except Exception:
        pass


def snapshot():
    env = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "power": "unknown", "battery": "", "charging": "", "mode": "", "cpu_load": "",
           "mem_avail_gb": "", "mem_total_gb": "", "mem_load_pct": "", "pagefile_mb": "", "omen_mode": omen_mode()}
    try:
        import psutil
        env["cpu_load"] = psutil.cpu_percent(interval=1.0)
    except Exception:
        pass
    try:
        _windows(env) if platform.system() == "Windows" else _linux(env)
    except Exception:
        pass
    return env


def describe(env):
    return (f"可用記憶體 {env['mem_avail_gb']} / {env['mem_total_gb']} GB、分頁檔 {env['pagefile_mb']} MB、"
            f"OMEN 模式 {env['omen_mode']}")


_started = None


def begin():
    global _started
    if _started is not None or os.environ.get("TIMING_ENV_LOG", "1") == "0":
        return
    _started = snapshot()
    print(f"計時環境(開始): 電源 {_started['power']}、電源模式 {_started['mode']}、{describe(_started)}", flush=True)
    atexit.register(_finish)


def _finish():
    end = snapshot()
    row = {"host": platform.node(), "script": Path(sys.argv[0]).name, "args": " ".join(sys.argv[1:])[:300],
           **{f"start_{k}": v for k, v in _started.items()}, **{f"end_{k}": v for k, v in end.items()}}
    try:
        grow = int(end["pagefile_mb"]) - int(_started["pagefile_mb"])
    except (TypeError, ValueError):
        grow = None
    row["pagefile_growth_mb"] = "" if grow is None else grow
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not LOG.exists()
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    except OSError:
        pass
    warn = "  ← 分頁檔增加超過 200 MB,這一輪可能有用到分頁檔,數字不能用" if grow is not None and grow > 200 else ""
    print(f"計時環境(結束): {describe(end)}{warn}", flush=True)
