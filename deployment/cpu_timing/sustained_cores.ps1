# 2026-11-15 核心排程對照(使用者要求;雙螢幕已排除,接著追「同一設定不同次開跑差 ±25%、頻率不變、兩個模型快慢會對調」是否為 E 核心排程)。
# 依序跑 sustained.ps1 的各模式,交錯排列以分開「模式的影響」和「每次開跑本來就有的差異」;每次先降溫再連續 180 秒。
# 用法(在這個資料夾,接電源、單螢幕、不要碰筆電,約 30–80 分鐘): .\sustained_cores.ps1
#   只跑部分: .\sustained_cores.ps1 -Plan any_3,pin_3
# 跑之前會先停掉背景的 cpu_probe.ps1(避免它插進來),跑完不會自動重開。
param([string[]]$Plan = @("any_1", "pin_1", "pcore_1", "any_2", "pin_2", "pcore_2", "any_3", "ecore_1"), [int]$Seconds = 180)
Set-Location $PSScriptRoot
Get-CimInstance Win32_Process -Filter "Name='pwsh.exe' OR Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'cpu_probe\.ps1' } |
    ForEach-Object { Write-Host "停掉背景探測 PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }
foreach ($t in $Plan) {
    $mode = ($t -split '_')[0]
    & .\sustained.ps1 -Tag "core_$t" -Mode $mode -Seconds $Seconds
}
Write-Host "$(Get-Date -Format 'HH:mm:ss') 全部跑完,結果在 compare_results\deploy_eval\sustained\report.md"
