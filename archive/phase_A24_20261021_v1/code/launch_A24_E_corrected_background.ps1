$ErrorActionPreference='Stop'
$a24Root=Split-Path -Parent $PSScriptRoot
$runRoot=Join-Path $a24Root 'recovery_E_wsl_20261023_v2'
$linuxCode='DRIVE_ROOT/wsl/2026-09-07/target-nature-e-you-are-a/unified_project/runs/phase_A24_20261021_v1/code'
$runtimeNode='DRIVE_ROOT/Users\USER_PROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
function Save-PipelineState($phase,$extra=@{}) {
    $value=@{time=[DateTimeOffset]::Now.ToString('o');phase=$phase;host_pid=$PID;R_N_rerun=$false;SPH_admitted=$false;max_fits=422}
    foreach($key in $extra.Keys){$value[$key]=$extra[$key]}
    $value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'background_pipeline.json') -Encoding utf8
}
try {
    if(Test-Path -LiteralPath (Join-Path $runRoot 'science_launch.json')){throw 'Corrective launch already exists; no retry'}
    Save-PipelineState 'corrective_E_starting_original_deadline'
    & wsl.exe -d Ubuntu-22.04 --cd / --exec /usr/bin/python3 -u -B "$linuxCode/execute_E_wsl_20261023_v2.py" --run --retain-original-deadline *> (Join-Path $runRoot 'supervisor.log')
    if($LASTEXITCODE -ne 0){throw 'Corrective WSL scientific supervisor failed; no retry'}
    $audit=Get-Content -LiteralPath (Join-Path $runRoot 'audit/completion.json') -Raw | ConvertFrom-Json
    $replay=Get-Content -LiteralPath (Join-Path $runRoot 'audit/checkpoint_replay.json') -Raw | ConvertFrom-Json
    if($audit.state -ne 'E_independent_zero_fit_audit_PASS' -or -not $audit.eligible_for_result_figure_generation -or $replay.state -ne 'PASS'){throw 'Figure gate did not pass'}
    Save-PipelineState 'result_gate_PASS_vector_figures_running'
    & wsl.exe -d Ubuntu-22.04 --cd / --exec /usr/bin/python3 -u -B "$linuxCode/build_A24_figures.py" *> (Join-Path $runRoot 'figure_builder.log')
    if($LASTEXITCODE -ne 0){throw 'Vector figure builder failed; no retry'}
    Save-PipelineState 'vector_figures_complete_editable_PPTX_running'
    & $runtimeNode (Join-Path $PSScriptRoot 'build_A24_editable.mjs') *> (Join-Path $runRoot 'pptx_builder.log')
    if($LASTEXITCODE -ne 0){throw 'PPTX builder failed; no retry'}
    Save-PipelineState 'artifacts_rendered_pending_visual_review' @{fits=422;new_figure_fits=0;visual_review_pending=$true;NC_submission_ready=$false}
} catch {
    Save-PipelineState 'stopped_no_retry' @{error=$_.Exception.Message}
    throw
}
