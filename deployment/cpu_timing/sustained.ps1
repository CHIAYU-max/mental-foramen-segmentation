# 2026-11-15 長時間負載對照: 雙螢幕 vs 單螢幕(使用者要求)。從這個資料夾執行;輸出在 repo 根目錄的 compare_results\。
# 每次: 先等降溫(溫度 ≤ -CoolC 且連續 60 秒,最多 -CoolMaxMinutes 分鐘)→ sustained_bench.py 連續跑 -Seconds 秒
#   (每圈第一階段 1 次 + U-Net INT8 5 次),同時每 2 秒記 CPU 實際效能 / 頻率 / 溫度 / 降頻比例 / 獨顯狀態 / 螢幕數。
# 用法(建議順序 單 → 雙 → 單 → 雙,換螢幕後再跑下一個):
#   .\sustained.ps1 -Tag single_1      .\sustained.ps1 -Tag dual_1      .\sustained.ps1 -Tag single_2      .\sustained.ps1 -Tag dual_2
# 11/15 加 -Mode any|pin|pcore|ecore(見 sustained_bench.py)與 P / E 核心各自的使用率(每 2 秒,原始計數差分);一次跑完全部: .\sustained_cores.ps1
# 輸出: compare_results\deploy_eval\sustained\<Tag>_calls.csv(每次呼叫)、<Tag>_sensors.csv(每 2 秒);彙整: python sustained_report.py
param([Parameter(Mandatory = $true)][string]$Tag, [int]$Seconds = 180, [int]$CoolC = 55, [int]$CoolMaxMinutes = 8, [ValidateSet("any", "pin", "pcore", "ecore")][string]$Mode = "any")
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }   # 用 $env:PYTHON 指定虛擬環境的 python.exe
$outDir = Join-Path $PSScriptRoot "..\..\compare_results\deploy_eval\sustained"
New-Item -ItemType Directory -Force $outDir | Out-Null
Add-Type -AssemblyName System.Windows.Forms

function CpuRaw { $h = @{}; Get-CimInstance Win32_PerfRawData_PerfOS_Processor | Where-Object { $_.Name -ne "_Total" } | ForEach-Object { $h[[int]$_.Name] = $_ }; $h }
function BusyPct($a, $b, $lps) { $v = foreach ($i in $lps) { 100 * (1 - ($b[$i].PercentProcessorTime - $a[$i].PercentProcessorTime) / [double]($b[$i].Timestamp_Sys100NS - $a[$i].Timestamp_Sys100NS)) }; [math]::Round(($v | Measure-Object -Average).Average, 1) }
$P_LPS = 0..7; $E_LPS = 8..15   # i5-13500H: 4 個 P 核心(含超執行緒)+ 8 個 E 核心,11/15 用 GetSystemCpuSetInformation 確認
function Temp { $tz = Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation -ErrorAction SilentlyContinue | Select-Object -First 1; if ($tz) { $tz.Temperature - 273 } else { -1 } }

$ps = [System.Windows.Forms.SystemInformation]::PowerStatus
if ($ps.PowerLineStatus -ne "Online") { Write-Host "沒有接電源,不測"; exit 2 }
$screens = [System.Windows.Forms.Screen]::AllScreens.Count
Write-Host "$(Get-Date -Format 'HH:mm:ss') ${Tag}(模式 $Mode): 螢幕 $screens 個,先等降溫到 ≤ $CoolC°C(連續 60 秒,最多 $CoolMaxMinutes 分鐘)"
$deadline = (Get-Date).AddMinutes($CoolMaxMinutes); $okSince = $null
while ((Get-Date) -lt $deadline) {
    $t = Temp
    if ($t -le $CoolC) { if (-not $okSince) { $okSince = Get-Date }; if (((Get-Date) - $okSince).TotalSeconds -ge 60) { break } } else { $okSince = $null }
    Start-Sleep -Seconds 5
}
$startTemp = Temp
Write-Host "$(Get-Date -Format 'HH:mm:ss') 開始(溫度 $startTemp°C),連續 $Seconds 秒,這段時間不要碰筆電"

$calls = Join-Path $outDir "${Tag}_calls.csv"
$sensors = Join-Path $outDir "${Tag}_sensors.csv"
$job = Start-Job -ScriptBlock { param($py, $dir, $out, $sec, $mode) Set-Location $dir; $env:PYTHONUTF8 = "1"; & $py sustained_bench.py $out --seconds $sec --mode $mode 2>&1 } -ArgumentList $py, $PSScriptRoot, $calls, $Seconds, $Mode
$rows = @()
$prev = CpuRaw
while ($job.State -eq "Running") {
    $now = CpuRaw; $pUtil = BusyPct $prev $now $P_LPS; $eUtil = BusyPct $prev $now $E_LPS; $prev = $now
    $pi = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'"
    $tz = Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation -ErrorAction SilentlyContinue | Select-Object -First 1
    $g = (& nvidia-smi --query-gpu=pstate,power.draw --format=csv,noheader,nounits 2>$null) -split ',' | ForEach-Object { $_.Trim() }
    $rows += [pscustomobject]@{ mode = $Mode; p_util = $pUtil; e_util = $eUtil; epoch = [math]::Round(([DateTimeOffset]::Now).ToUnixTimeMilliseconds() / 1000.0, 3); perf_pct = $pi.PercentProcessorPerformance
        freq_mhz = $pi.ProcessorFrequency; temp_c = if ($tz) { $tz.Temperature - 273 } else { "" }; passive_limit = if ($tz) { $tz.PercentPassiveLimit } else { "" }
        dgpu_pstate = $g[0]; dgpu_power_w = $g[1]; screens = [System.Windows.Forms.Screen]::AllScreens.Count; ac = [int]([System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus -eq "Online") }
    Start-Sleep -Seconds 2
}
$out = (Receive-Job $job) -join " "; Remove-Job $job
$rows | Export-Csv $sensors -Encoding utf8
Write-Host "$(Get-Date -Format 'HH:mm:ss') 結束(溫度 $(Temp)°C)$(if ($out -notmatch 'done') { ";基準輸出: $out" } else { ",$($out -replace '.*(done.*)', '$1')" })"
& $py sustained_report.py

