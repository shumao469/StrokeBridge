param([Parameter(Mandatory=$true)][string]$RunRoot,[Parameter(Mandatory=$true)][int]$WorkerPid)
$ErrorActionPreference='Stop'
$needle='*execute_E_cpu8h_once.py --module extract_controls.py*'
$worker=Get-CimInstance Win32_Process -Filter "ProcessId = $WorkerPid"
if ($null -eq $worker) { exit 0 }
if ($worker.CommandLine -notlike $needle) { throw 'Worker identity mismatch; no action' }
$created=[DateTimeOffset]$worker.CreationDate
$deadline=$created.AddSeconds(180)
$receipt=Join-Path $RunRoot 'preflight_deadline_guard.json'
function Save-Guard([string]$state) { [ordered]@{time=[DateTimeOffset]::Now.ToString('o');state=$state;worker_pid=$WorkerPid;worker_created=$created.ToString('o');deadline=$deadline.ToString('o');guard_pid=$PID;fits=0;inference_calls=0} | ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding UTF8 }
Save-Guard 'monitoring_own_preflight_only'
while ($true) {
    $worker=Get-CimInstance Win32_Process -Filter "ProcessId = $WorkerPid"
    if ($null -eq $worker) { Save-Guard 'worker_exited_no_action'; exit 0 }
    if ($worker.CommandLine -notlike $needle -or [Math]::Abs((([DateTimeOffset]$worker.CreationDate)-$created).TotalSeconds) -gt 1) { Save-Guard 'identity_changed_no_action'; exit 1 }
    $benchFile=Join-Path $RunRoot 'extraction/CPU_benchmark.json'
    if (Test-Path -LiteralPath $benchFile) {
        $bench=Get-Content -LiteralPath $benchFile -Raw | ConvertFrom-Json
        if ($bench.admitted_for_bulk -eq $true) { Save-Guard 'preflight_admitted_parent_enforces8hour_cap'; exit 0 }
    }
    if ([DateTimeOffset]::Now -ge $deadline) { Save-Guard 'preflight_deadline_terminating_own_child'; Stop-Process -Id $WorkerPid; Save-Guard 'own_preflight_child_stopped_no_retry'; exit 2 }
    Start-Sleep -Seconds 2
}
