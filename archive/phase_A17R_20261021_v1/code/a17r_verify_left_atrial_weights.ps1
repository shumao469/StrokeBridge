$ErrorActionPreference='Stop'
$taskRun='DRIVE_ROOT/wsl\2026-09-07\target-nature-e-you-are-a\unified_project\runs\phase_A17R_20261021_v1'
$taskAudit=Join-Path $taskRun 'left_atrial_source_audit'
$taskRevision='285eb9db4fa85ca164f96311be590c99ab7e0e79'
$taskDestination="DRIVE_ROOT/脑心同查\models\ecg_ai_atrialmeasures\$taskRevision"
$taskMeta=Get-Content -LiteralPath (Join-Path $taskAudit 'huggingface_model.json') -Raw | ConvertFrom-Json
if($taskMeta.sha -ne $taskRevision){throw 'Provider revision mismatch'}
$taskFiles=@($taskMeta.siblings | Where-Object {$_.rfilename -match '^(LAEF|LAVmax|LAVmin)_[0-4]\.h5$'})
if($taskFiles.Count -ne 15){throw 'Expected 15 weights'}
$taskChecks=@();$taskTotal=0L
foreach($taskFile in $taskFiles){
    $taskPath=Join-Path $taskDestination $taskFile.rfilename
    $taskSize=(Get-Item -LiteralPath $taskPath).Length
    $taskHash=(Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if($taskSize -ne $taskFile.size -or $taskHash -ne $taskFile.lfs.sha256){throw "Integrity failure: $($taskFile.rfilename)"}
    $taskStream=[IO.File]::OpenRead($taskPath)
    try{
        $taskBytes=[byte[]]::new(8)
        if($taskStream.Read($taskBytes,0,8) -ne 8 -or [BitConverter]::ToString($taskBytes) -ne '89-48-44-46-0D-0A-1A-0A'){throw "HDF5 signature failure: $($taskFile.rfilename)"}
    }finally{$taskStream.Dispose()}
    $taskChecks+=[ordered]@{file=$taskFile.rfilename;bytes=$taskSize;sha256=$taskHash;size_SHA256_HDF5_signature='PASS'}
    $taskTotal+=$taskSize
}
$taskReceipt=[ordered]@{verified_at=(Get-Date).ToString('o');revision=$taskRevision;verified_files=$taskChecks.Count;verified_bytes=$taskTotal;state='independent_size_SHA256_HDF5_signature_PASS';files=$taskChecks;model_deserialization_verified=$false;inference_started=$false;clinical_accuracy_validated=$false;classifier_fits_added=0}
$taskReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskAudit 'weights_independent_verification.json') -Encoding utf8
$taskReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskDestination 'independent_verification.json') -Encoding utf8
[ordered]@{verified_files=$taskChecks.Count;verified_bytes=$taskTotal;state=$taskReceipt.state} | ConvertTo-Json -Compress
