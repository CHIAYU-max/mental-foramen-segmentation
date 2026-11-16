# 2026-11-15 筆電 CPU 速度探測(追「同一模型在不同時段差到 1.7 倍」的原因,使用者要求)。從這個資料夾執行;輸出在 repo 根目錄的 compare_results\。
# 每 -IntervalMinutes 分鐘跑一次 cpu_probe_bench.py(約 15 秒,隨機輸入,不需要病人影像),同時記錄:
#   電源 / 電量 / 是否充電 / 電源模式、開始前 CPU 使用率、ACPI 溫度區溫度與降頻比例(前後各一次)、
#   基準進行中的實際效能百分比(相對基頻 2.6 GHz,每秒取樣)、多久沒有鍵盤滑鼠輸入、前景視窗的程式。
# 用法: .\cpu_probe.ps1                       每 10 分鐘一次,跑 24 小時
#       .\cpu_probe.ps1 -Once -Tag omen_perf   只跑一次並標記(例: 在 OMEN Gaming Hub 切換模式後各跑一次)
# 輸出: compare_results\deploy_eval\cpu_probe\probe.csv(每次一列,一直往下加)
param([int]$IntervalMinutes = 10, [double]$Hours = 24, [switch]$Once, [string]$Tag = "")
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }   # 用 $env:PYTHON 指定虛擬環境的 python.exe
$env:PYTHONPATH = (Resolve-Path "$PSScriptRoot\..\..").Path   # 讓 python 找得到 deployment.no_throttle
$outDir = Join-Path $PSScriptRoot "..\..\compare_results\deploy_eval\cpu_probe"
New-Item -ItemType Directory -Force $outDir | Out-Null
$csv = Join-Path $outDir "probe_v2.csv"   # 11/15 加了螢幕數、亮度、獨顯狀態;欄位不同,另開新檔(舊的 probe.csv 保留)
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System; using System.Runtime.InteropServices; using System.Text;
public static class ProbeWin {
    [StructLayout(LayoutKind.Sequential)] public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
    [DllImport("user32.dll")] public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("kernel32.dll")] public static extern uint GetTickCount();
    public static double IdleSeconds() { var i = new LASTINPUTINFO(); i.cbSize = (uint)Marshal.SizeOf(i); GetLastInputInfo(ref i); return unchecked(GetTickCount() - i.dwTime) / 1000.0; }
    public static uint ForegroundPid() { uint pid; GetWindowThreadProcessId(GetForegroundWindow(), out pid); return pid; }
}
"@

function Get-Power {
    $ps = [System.Windows.Forms.SystemInformation]::PowerStatus
    $ac = $ps.PowerLineStatus -eq "Online"
    $overlays = @{ "ded574b5-45a0-4f42-8737-46345c09c238" = "best_performance"; "961cc777-2547-4f9d-8174-7d86181b8a7a" = "best_efficiency";
                   "00000000-0000-0000-0000-000000000000" = "balanced" }
    $mode = ""
    try {
        $key = Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
        $guid = "$(if ($ac) { $key.ActiveOverlayAcPowerScheme } else { $key.ActiveOverlayDcPowerScheme })".ToLower()
        $mode = if ($overlays.ContainsKey($guid)) { $overlays[$guid] } else { $guid }
    } catch { }
    [pscustomobject]@{ ac = [int]$ac; battery = [int]($ps.BatteryLifePercent * 100);
                       charging = [int][bool]($ps.BatteryChargeStatus -band [System.Windows.Forms.BatteryChargeStatus]::Charging); mode = $mode }
}

function Get-CpuLoad([int]$seconds = 5) {
    $raw = { Get-CimInstance Win32_PerfRawData_PerfOS_Processor -Filter "Name='_Total'" }
    $a = & $raw; Start-Sleep -Seconds $seconds; $b = & $raw
    [math]::Round(100 * (1 - ($b.PercentProcessorTime - $a.PercentProcessorTime) / [double]($b.Timestamp_Sys100NS - $a.Timestamp_Sys100NS)), 1)
}

function Get-Thermal {
    $tz = Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($tz) { [pscustomobject]@{ temp_c = $tz.Temperature - 273; passive_limit = $tz.PercentPassiveLimit; throttle_reasons = $tz.ThrottleReasons } }
    else { [pscustomobject]@{ temp_c = ""; passive_limit = ""; throttle_reasons = "" } }
}

function Probe {
    $p = Get-Power
    $idle = [math]::Round([ProbeWin]::IdleSeconds())
    $fgPid = [ProbeWin]::ForegroundPid()
    $fg = (Get-Process -Id $fgPid -ErrorAction SilentlyContinue).ProcessName
    # 11/15 使用者的推測: 接雙螢幕(外接輸出多半接在獨顯上,獨顯無法休眠)與螢幕亮度可能拖慢 CPU → 一起記
    $screens = [System.Windows.Forms.Screen]::AllScreens.Count
    $bright = try { (Get-CimInstance -Namespace root/wmi WmiMonitorBrightness -ErrorAction Stop | Select-Object -First 1).CurrentBrightness } catch { "" }
    $gpuIdle = (& nvidia-smi --query-gpu=pstate,power.draw --format=csv,noheader,nounits 2>$null) -split ',' | ForEach-Object { $_.Trim() }
    $load = Get-CpuLoad 5
    $t0 = Get-Thermal
    # 基準在背景跑,這邊每秒取一次實際效能百分比與頻率(Processor Information 類別,CIM 名稱不受語系影響)
    $job = Start-Job -ScriptBlock { param($py, $dir) Set-Location $dir; $env:PYTHONUTF8 = "1"; & $py cpu_probe_bench.py 15 2>&1 } -ArgumentList $py, $PSScriptRoot
    $perf = @(); $freq = @()
    Start-Sleep -Seconds 3   # 跳過載入模型的時間
    while ($job.State -eq "Running") {
        $pi = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'"
        $perf += $pi.PercentProcessorPerformance; $freq += $pi.ProcessorFrequency
        Start-Sleep -Seconds 1
    }
    $outText = (Receive-Job $job) -join "`n"; Remove-Job $job
    $t1 = Get-Thermal
    $bench = $null
    try { $bench = ($outText -split "`n" | Where-Object { $_ -match '^\{' } | Select-Object -Last 1) | ConvertFrom-Json } catch { }
    $row = [ordered]@{
        time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"; tag = $Tag; ac = $p.ac; battery = $p.battery; charging = $p.charging; mode = $p.mode
        idle_input_s = $idle; foreground = $fg; cpu_load_before = $load
        screens = $screens; brightness = $bright; dgpu_pstate = $gpuIdle[0]; dgpu_power_w = $gpuIdle[1]
        temp_before_c = $t0.temp_c; temp_after_c = $t1.temp_c; passive_limit_after = $t1.passive_limit; throttle_reasons_after = $t1.throttle_reasons
        perf_pct_under_load = if ($perf) { [math]::Round(($perf | Measure-Object -Average).Average, 1) } else { "" }
        perf_pct_min = if ($perf) { ($perf | Measure-Object -Minimum).Minimum } else { "" }
        freq_mhz_under_load = if ($freq) { [math]::Round(($freq | Measure-Object -Average).Average) } else { "" }
        unet_int8_ms = $bench.unet_int8_ms; unet_int8_p90_ms = $bench.unet_int8_p90_ms; stage1_ms = $bench.stage1_ms; stage1_p90_ms = $bench.stage1_p90_ms
        error = if ($bench) { "" } else { ($outText -replace "[\r\n,]+", " ").Substring(0, [math]::Min(200, $outText.Length)) }
    }
    [pscustomobject]$row | Export-Csv $csv -Append -Encoding utf8
    Write-Host ("{0} U-Net INT8 {1} ms、第一階段 {2} ms、效能 {3}%({4} MHz)、溫度 {5}→{6}°C、電源 {7}/{8}%/{9}、螢幕 {12} 個、亮度 {13}、獨顯 {14} {15} W、閒置 {10} s、前景 {11}" -f
        $row.time, $row.unet_int8_ms, $row.stage1_ms, $row.perf_pct_under_load, $row.freq_mhz_under_load, $row.temp_before_c, $row.temp_after_c,
        $(if ($p.ac) { "AC" } else { "電池" }), $p.battery, $p.mode, $idle, $fg, $screens, $bright, $gpuIdle[0], $gpuIdle[1])
}

if ($Once) { Probe; exit 0 }
$end = (Get-Date).AddHours($Hours)
while ((Get-Date) -lt $end) {
    $next = (Get-Date).AddMinutes($IntervalMinutes)
    Probe
    $wait = ($next - (Get-Date)).TotalSeconds
    if ($wait -gt 0) { Start-Sleep -Seconds $wait }
}
