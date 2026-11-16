# 2026-11-16 驗證: 背景探測整夜第一階段 330–345 ms(前景是探測自己的視窗時 200–225 ms)→ 是否為 Windows 對非前景程式的 power throttling(EcoQoS)。
# 三種方式各 -Rounds 次、交錯: fg = 從目前的終端機直接跑(前景);bg = 最小化的新視窗(非前景);bg_nothrottle = 同 bg,但 python 用
# SetProcessInformation(ProcessPowerThrottling) 關掉 EXECUTION_SPEED 節流。每次 cpu_probe_bench.py 15 秒,隨機輸入。
# 輸出: compare_results\deploy_eval\cpu_probe\throttle_test.csv
param([int]$Rounds = 2)
Set-Location $PSScriptRoot
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }   # 用 $env:PYTHON 指定虛擬環境的 python.exe
$env:PYTHONPATH = (Resolve-Path "$PSScriptRoot\..\..").Path   # 讓 python 找得到 deployment.no_throttle
$out = Join-Path $PSScriptRoot "..\..\compare_results\deploy_eval\cpu_probe\throttle_test.csv"
$tmp = Join-Path $env:TEMP "throttle_test_out.txt"
Add-Type @"
using System; using System.Runtime.InteropServices;
public static class FgWin { [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  public static uint Pid() { uint p; GetWindowThreadProcessId(GetForegroundWindow(), out p); return p; } }
"@
$opt = "import ctypes, runpy, sys; from ctypes import wintypes as w`n" +
       "class S(ctypes.Structure): _fields_ = [('Version', w.ULONG), ('ControlMask', w.ULONG), ('StateMask', w.ULONG)]`n" +
       "s = S(1, 1, 0); ok = ctypes.windll.kernel32.SetProcessInformation(ctypes.windll.kernel32.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s))`n" +
       "print('throttle_off', ok); sys.argv = ['cpu_probe_bench.py', '15']; runpy.run_path('cpu_probe_bench.py', run_name='__main__')"
$optFile = Join-Path $env:TEMP "throttle_off.py"; Set-Content $optFile $opt -Encoding utf8
$rows = @()
for ($r = 1; $r -le $Rounds; $r++) {
    $order = if ($r % 2) { "fg", "bg", "bg_nothrottle" } else { "bg_nothrottle", "bg", "fg" }
    foreach ($k in $order) {
        Start-Sleep -Seconds 20   # 讓溫度回落
        $fg = (Get-Process -Id ([FgWin]::Pid()) -ErrorAction SilentlyContinue).ProcessName
        if ($k -eq "fg") { $text = (& $py cpu_probe_bench.py 15 2>&1) -join "`n" }
        else {
            $args2 = if ($k -eq "bg") { "cpu_probe_bench.py 15" } else { "`"$optFile`"" }
            $p = Start-Process pwsh -WindowStyle Minimized -PassThru -ArgumentList "-NoProfile", "-Command",
                "Set-Location '$PSScriptRoot'; `$env:PYTHONUTF8='1'; & '$py' $args2 *> '$tmp'"
            $p.WaitForExit(); $text = Get-Content $tmp -Raw
        }
        $j = ($text -split "`n" | Where-Object { $_ -match '^\{' } | Select-Object -Last 1) | ConvertFrom-Json
        $row = [pscustomobject]@{ time = Get-Date -Format "HH:mm:ss"; kind = $k; round = $r; foreground = $fg; stage1_ms = $j.stage1_ms; unet_int8_ms = $j.unet_int8_ms
                                  throttle_off = if ($text -match 'throttle_off (\d)') { $Matches[1] } else { "" } }
        $rows += $row; Write-Host ($row | ConvertTo-Json -Compress)
    }
}
$rows | Export-Csv $out -Encoding utf8
