param([Parameter(Mandatory=$true)][string]$RecoveryRoot)
$ErrorActionPreference = 'Stop'
$receiptPath = Join-Path $RecoveryRoot 'deadline_guard.json'
$launch = Get-Content -LiteralPath (Join-Path $RecoveryRoot 'launch.json') -Raw | ConvertFrom-Json
$expectedPid = [int]$launch.pid
$started = [DateTimeOffset]::Parse($launch.time)
$deadline = $started.AddSeconds([int]$launch.max_total_seconds)
function Write-Guard([string]$state, [string]$detail) {
    [ordered]@{time=[DateTimeOffset]::Now.ToString('o');state=$state;detail=$detail;worker_pid=$expectedPid;deadline=$deadline.ToString('o');guard_pid=$PID;inference_calls=0;fits=0} |
        ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding UTF8
}
Write-Guard 'monitoring_own_worker_only' 'External wall-time enforcement; no scientific or frozen-code changes.'
while ($true) {
    $worker = Get-CimInstance Win32_Process -Filter "ProcessId = $expectedPid"
    if ($null -eq $worker) { Write-Guard 'worker_exited' 'No process terminated.'; exit 0 }
    if ($worker.CommandLine -notlike '*recover_N_once.py*' -or [Math]::Abs((([DateTimeOffset]$worker.CreationDate)-$started).TotalSeconds) -gt 5) {
        Write-Guard 'identity_mismatch_no_action' 'PID/command/creation time did not match; no process terminated.'; exit 1
    }
    if ([DateTimeOffset]::Now -ge $deadline) {
        Write-Guard 'deadline_reached_terminating_own_worker' 'Exactly scoped to the launch PID and creation time; no service/other job changes.'
        Stop-Process -Id $expectedPid -ErrorAction Stop
        Write-Guard 'own_worker_stopped_at_deadline' 'Recovery cannot resume automatically; retain partial outputs and original receipts.'
        exit 2
    }
    Start-Sleep -Seconds 5
}
