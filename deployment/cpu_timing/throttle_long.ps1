# 2026-11-16 驗證(接 throttle_test.ps1): 在最小化的背景視窗待久了,第一階段是否變慢(Windows 對背景程式的 power throttling / EcoQoS)。
# 這支腳本本身要用最小化視窗啟動;每 -EveryMin 分鐘跑一次 cpu_probe_bench.py(15 秒),交替 normal / nothrottle
# (nothrottle = python 用 SetProcessInformation(ProcessPowerThrottling, ControlMask=EXECUTION_SPEED, StateMask=0) 明確關掉節流)。
# 輸出: compare_results\deploy_eval\cpu_probe\throttle_long.csv(每次一列,一直往下加)
param([int]$Count = 9, [int]$EveryMin = 4)
Set-Location $PSScriptRoot
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }   # 用 $env:PYTHON 指定虛擬環境的 python.exe
$env:PYTHONPATH = (Resolve-Path "$PSScriptRoot\..\..").Path   # 讓 python 找得到 deployment.no_throttle
$out = Join-Path $PSScriptRoot "..\..\compare_results\deploy_eval\cpu_probe\throttle_long.csv"
$optFile = Join-Path $env:TEMP "throttle_off2.py"
@"
import ctypes, runpy, sys
from ctypes import wintypes as w
class S(ctypes.Structure):
    _fields_ = [("Version", w.ULONG), ("ControlMask", w.ULONG), ("StateMask", w.ULONG)]
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.GetCurrentProcess.restype = w.HANDLE
k.SetProcessInformation.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
s = S(1, 1, 0)
ok = k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s))
print("throttle_off", int(bool(ok)), ctypes.get_last_error())
sys.argv = ["cpu_probe_bench.py", "15"]
runpy.run_path("cpu_probe_bench.py", run_name="__main__")
"@ | Set-Content $optFile -Encoding utf8
$t0 = Get-Date
for ($i = 0; $i -lt $Count; $i++) {
    $next = $t0.AddMinutes($EveryMin * ($i + 1))
    $kind = if ($i % 2) { "nothrottle" } else { "normal" }
    $env:PYTHONUTF8 = "1"
    $text = if ($kind -eq "normal") { (& $py cpu_probe_bench.py 15 2>&1) -join "`n" } else { (& $py $optFile 2>&1) -join "`n" }
    $j = ($text -split "`n" | Where-Object { $_ -match '^\{' } | Select-Object -Last 1) | ConvertFrom-Json
    [pscustomobject]@{ time = Get-Date -Format "HH:mm:ss"; minutes_in_bg = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1); kind = $kind
                       throttle_off = if ($text -match 'throttle_off (\d)') { $Matches[1] } else { "" }; stage1_ms = $j.stage1_ms; unet_int8_ms = $j.unet_int8_ms } |
        Export-Csv $out -Append -Encoding utf8
    $wait = ($next - (Get-Date)).TotalSeconds; if ($wait -gt 0 -and $i -lt $Count - 1) { Start-Sleep -Seconds $wait }
}
