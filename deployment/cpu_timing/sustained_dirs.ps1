# 2026-11-15 兩個新方向(使用者要求;接在核心排程對照之後): 筆電跨時段延遲變異來自混合架構的執行緒排程,
#   1. Windows 電源計畫的異質執行緒排程原則 SCHEDPOLICY(預設 5 = Automatic)改成 2 = Prefer performant processors
#   2. 第一階段 OpenVINO 執行緒 12 → 8(sustained_bench.py --mode t8;約 4 P + 4 E,U-Net 維持 4)
# 交錯跑 base(原則 5、預設)/ t8(原則 5、8 執行緒)/ pp(原則 2、預設),每種 -Rounds 次,每次先降溫再連續 180 秒(sustained.ps1)。
# 原則只改目前電源計畫的 AC 值,結束(含中斷)一律還原成原本的值;不需要系統管理員。
# 用法(在這個資料夾,接電源、不要碰筆電,約 3 輪 × 3 × 4–11 分鐘): .\sustained_dirs.ps1 [-Rounds 3]
# 跑之前停掉背景的 cpu_probe.ps1;跑完不自動重開。結果: compare_results\deploy_eval\sustained\report.md(Tag dir_<種類>_<輪>)。
param([int]$Rounds = 3, [int]$Seconds = 180)
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "sustained_dirs.log"
function Log($m) { $line = "$(Get-Date -Format 'HH:mm:ss') $m"; Write-Host $line; Add-Content $log $line }
function Get-Policy { [Convert]::ToInt32((((powercfg /qh scheme_current sub_processor SCHEDPOLICY) | Select-String "Current AC").Line -split ':')[1].Trim(), 16) }
function Set-Policy([int]$v) { powercfg /setacvalueindex scheme_current sub_processor SCHEDPOLICY $v | Out-Null; powercfg /setactive scheme_current | Out-Null; Get-Policy }

Get-CimInstance Win32_Process -Filter "Name='pwsh.exe' OR Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'cpu_probe\.ps1' } |
    ForEach-Object { Log "停掉背景探測 PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }
$orig = Get-Policy
Log "開始: 原本的 SCHEDPOLICY(AC)= $orig"
$kinds = @(@{ name = "base"; policy = $orig; mode = "any" }, @{ name = "t8"; policy = $orig; mode = "t8" }, @{ name = "pp"; policy = 2; mode = "any" })
try {
    for ($r = 1; $r -le $Rounds; $r++) {
        $order = if ($r % 2) { $kinds } else { $kinds[2], $kinds[0], $kinds[1] }   # 每輪換順序
        foreach ($k in $order) {
            $now = Set-Policy $k.policy
            Log "dir_$($k.name)_${r}: SCHEDPOLICY=$now, mode=$($k.mode)"
            & .\sustained.ps1 -Tag "dir_$($k.name)_$r" -Mode $k.mode -Seconds $Seconds
        }
    }
}
finally {
    $back = Set-Policy $orig
    Log "結束: SCHEDPOLICY 還原為 $back(原本 $orig)"
    Add-Content $log "SUSTAINED_DIRS_EXIT=$(if ($back -eq $orig) { 0 } else { 1 })"
}
