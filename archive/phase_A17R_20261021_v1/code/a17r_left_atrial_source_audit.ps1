$ErrorActionPreference='Stop'
$target='DRIVE_ROOT/wsl\2026-09-07\target-nature-e-you-are-a\unified_project\runs\phase_A17R_20261021_v1\left_atrial_source_audit'
New-Item -ItemType Directory -Path $target -Force | Out-Null
$headers=@{'User-Agent'='BrainHeartResearch-A17R-source-audit';'Accept'='application/json'}
$sources=@{
    github_repository='https://api.github.com/repos/j-brody/ecg_ai_atrialmeasures'
    github_license='https://api.github.com/repos/j-brody/ecg_ai_atrialmeasures/license'
    github_revision='https://api.github.com/repos/j-brody/ecg_ai_atrialmeasures/commits/main'
    huggingface_model='https://huggingface.co/api/models/j-brody/ecg_ai_atrialmeasures-models?blobs=true'
    zenodo_software='https://zenodo.org/api/records/20597014'
}
$results=@{}
foreach($name in $sources.Keys){
    try {
        $response=Invoke-WebRequest -Uri $sources[$name] -Headers $headers -TimeoutSec 25
        $payload=$response.Content | ConvertFrom-Json
        $payload | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath (Join-Path $target "$name.json") -Encoding utf8
        $results[$name]=[ordered]@{url=$sources[$name];http_status=[int]$response.StatusCode;metadata_saved=$true}
    } catch {
        $results[$name]=[ordered]@{url=$sources[$name];http_status=if($_.Exception.Response){[int]$_.Exception.Response.StatusCode}else{$null};metadata_saved=$false;error=$_.Exception.Message}
    }
}
$hf=$null;$repo=$null;$zenodo=$null
if(Test-Path (Join-Path $target 'huggingface_model.json')){$hf=Get-Content (Join-Path $target 'huggingface_model.json') -Raw | ConvertFrom-Json}
if(Test-Path (Join-Path $target 'github_repository.json')){$repo=Get-Content (Join-Path $target 'github_repository.json') -Raw | ConvertFrom-Json}
if(Test-Path (Join-Path $target 'zenodo_software.json')){$zenodo=Get-Content (Join-Path $target 'zenodo_software.json') -Raw | ConvertFrom-Json}
$summary=[ordered]@{
    audit_date='2026-10-21';sources=$results
    code_license=if($repo){$repo.license}else{$null}
    weights_license=if($hf){$hf.cardData.license}else{$null}
    weights_revision=if($hf){$hf.sha}else{$null}
    weights_files=if($hf){@($hf.siblings | Select-Object rfilename,lfs,size)}else{@()}
    software_archive_license=if($zenodo){$zenodo.metadata.license}else{$null}
    expected_input='500Hz median beat, 600 samples, leads I II V1 V2 V3 V4 V5 V6; verify exact units/scaling from fixed official code before inference'
    status='source_audit_only_pending_explicit_licence_and_input_contract_review'
    weight_downloaded=$false;inference_started=$false;included_in_28_fit_comparison=$false
    ground_truth='No local paired left-atrial imaging truth available; no accuracy or clinical claim'
}
$summary | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $target 'audit_summary.json') -Encoding utf8
[ordered]@{sources=$results;code_license=$summary.code_license;weights_license=$summary.weights_license;revision=$summary.weights_revision;weight_files=$summary.weights_files.Count;status=$summary.status} | ConvertTo-Json -Depth 5
