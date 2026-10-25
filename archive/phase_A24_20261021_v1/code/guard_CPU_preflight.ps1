param([string]$RunRoot,[int]$WorkerPid=37128)
$ErrorActionPreference='Stop'
$worker=Get-CimInstance Win32_Process -Filter "ProcessId = $WorkerPid"
if ($null -eq $worker) { exit 0 }
if ($worker.CommandLine -notlike '*execute_E_cpu_once.py --module extract_controls.py*') { throw 'Worker identity mismatch: no action' }
$start=[DateTimeOffset]$worker.CreationDate
$deadline=$start.AddSeconds(180)
$receipt=Join-Path $RunRoot 'preflight_deadline_guard.json'
function Write-Receipt([string]$state) {
    [ordered]@{time=[DateTimeOffset]::Now.ToString('o');state=$state;worker_pid=$WorkerPid;worker_created=$start.ToString('o');deadline=$deadline.ToString('o');guard_pid=$PID;fits=0;inference_calls=0} | ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding UTF8
}
Write-Receipt 'monitoring_own_preflight_only'
while ($true) {
    $worker=Get-CimInstance Win32_Process -Filter "ProcessId = $WorkerPid"
    if ($null -eq $worker) { Write-Receipt 'worker_exited_no_action'; exit 0 }
    if ($worker.CommandLine -notlike '*execute_E_cpu_once.py --module extract_controls.py*' -or [Math]::Abs((([DateTimeOffset]$worker.CreationDate)-$start).TotalSeconds) -gt 1) { Write-Receipt 'identity_changed_no_action'; exit 1 }
    $admission=Join-Path $RunRoot 'extraction/CPU_benchmark.json'
    if (Test-Path -LiteralPath $admission) {
        $bench=Get-Content -LiteralPath $admission -Raw | ConvertFrom-Json
        if ($bench.admitted_for_bulk -eq $true) { Write-Receipt 'preflight_admitted_parent_keeps7200second_cap'; exit 0 }
    }
    if ([DateTimeOffset]::Now -ge $deadline) {
        Write-Receipt '180second_preflight_limit_terminating_own_child'
        Stop-Process -Id $WorkerPid
        Write-Receipt 'own_preflight_child_stopped_no_retry'
        exit 2
    }
    Start-Sleep -Seconds 2
}
