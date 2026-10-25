$ErrorActionPreference='Stop'
$run='DRIVE_ROOT/wsl\2026-09-07\target-nature-e-you-are-a\unified_project\runs\phase_A17R_20261021_v1'
$audit=Join-Path $run 'left_atrial_source_audit'
$meta=Get-Content -LiteralPath (Join-Path $audit 'huggingface_model.json') -Raw | ConvertFrom-Json
$revision='285eb9db4fa85ca164f96311be590c99ab7e0e79'
if($meta.sha -ne $revision -or $meta.cardData.license -ne 'mit'){throw 'Pinned revision or model-card licence mismatch'}
$destination="DRIVE_ROOT/脑心同查\models\ecg_ai_atrialmeasures\$revision"
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$files=@($meta.siblings | Where-Object {$_.rfilename -like '*.h5'})
if($files.Count -ne 15){throw 'Expected five weights for each of three phenotypes'}
foreach($file in $files){if($file.rfilename -notmatch '^(LAVmin|LAVmax|LAEF)_[0-4]\.h5$' -or !$file.lfs.sha256 -or $file.size -ne $file.lfs.size){throw 'Unexpected filename/size/SHA contract'}}
Copy-Item -LiteralPath (Join-Path $audit 'huggingface_model.json') -Destination (Join-Path $destination 'provider_model_metadata.json') -Force
Copy-Item -LiteralPath (Join-Path $audit '来源许可与输入审查结论.txt') -Destination (Join-Path $destination '来源许可与输入审查结论.txt') -Force
$client=[System.Net.Http.HttpClient]::new()
$client.Timeout=[TimeSpan]::FromMinutes(15)
$client.DefaultRequestHeaders.UserAgent.ParseAdd('BrainHeartResearch-A17R/1.0')
$items=@()
$started=Get-Date
function Save-Receipt([string]$State){
    $totalVerified=0L
    foreach($entry in $items){$totalVerified += [long]$entry['bytes']}
    [ordered]@{state=$State;started_at=$started.ToString('o');updated_at=(Get-Date).ToString('o');source='https://huggingface.co/j-brody/ecg_ai_atrialmeasures-models';revision=$revision;model_card_license='mit';destination=$destination;expected_files=15;verified_files=$items.Count;verified_bytes=$totalVerified;files=$items;inference_started=$false;clinical_accuracy_validated=$false;included_in_A17R_seven_arms=$false;classifier_fits_added=0} | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $destination 'verification.json') -Encoding utf8
}
try{
 foreach($file in $files){
    $target=Join-Path $destination $file.rfilename
    $partial=$target+'.partial'
    if(Test-Path -LiteralPath $target){
        $h=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower()
        if((Get-Item -LiteralPath $target).Length -ne $file.size -or $h -ne $file.lfs.sha256){throw "Existing final weight fails integrity: $($file.rfilename)"}
    }else{
        for($attempt=1;$attempt -le 2;$attempt++){
            $response=$null;$inputStream=$null;$outputStream=$null;$request=$null
            try{
                $offset=if(Test-Path -LiteralPath $partial){(Get-Item -LiteralPath $partial).Length}else{0L}
                if($offset -gt $file.size){throw 'Partial larger than provider expected size'}
                if($offset -lt $file.size){
                    $url="https://huggingface.co/j-brody/ecg_ai_atrialmeasures-models/resolve/$revision/$($file.rfilename)?download=true"
                    $request=[System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Get,$url)
                    if($offset -gt 0){$request.Headers.Range=[System.Net.Http.Headers.RangeHeaderValue]::new($offset,$null)}
                    $response=$client.SendAsync($request,[System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
                    $null=$response.EnsureSuccessStatusCode()
                    if($offset -gt 0 -and ([int]$response.StatusCode -ne 206 -or $response.Content.Headers.ContentRange.From -ne $offset)){throw 'Resume range not honoured; partial preserved'}
                    if($offset -eq 0 -and [int]$response.StatusCode -ne 200){throw 'Unexpected initial response'}
                    $inputStream=$response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
                    $outputStream=[IO.FileStream]::new($partial,[IO.FileMode]::Append,[IO.FileAccess]::Write,[IO.FileShare]::Read,1048576)
                    $buffer=[byte[]]::new(1048576);$received=$offset;$timer=[Diagnostics.Stopwatch]::StartNew();$last=0
                    while(($n=$inputStream.Read($buffer,0,$buffer.Length)) -gt 0){
                        $outputStream.Write($buffer,0,$n);$received+=$n
                        if($received -gt $file.size){throw 'Server payload exceeds provider size'}
                        if($timer.Elapsed.TotalSeconds-$last -ge 10){$last=$timer.Elapsed.TotalSeconds;[ordered]@{file=$file.rfilename;received_bytes=$received;expected_bytes=$file.size;MiB_per_s=[math]::Round(($received-$offset)/1048576/[math]::Max($timer.Elapsed.TotalSeconds,.001),2)} | ConvertTo-Json -Compress | Write-Output}
                    }
                    $outputStream.Flush($true);$outputStream.Dispose();$outputStream=$null
                }
                if((Get-Item -LiteralPath $partial).Length -ne $file.size){throw 'Incomplete transfer; partial preserved'}
                $h=(Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLower()
                if($h -ne $file.lfs.sha256){throw 'LFS SHA256 mismatch; partial preserved'}
                Move-Item -LiteralPath $partial -Destination $target
                break
            }catch{
                [ordered]@{file=$file.rfilename;attempt=$attempt;state='transfer_failed_partial_preserved';reason=$_.Exception.Message} | ConvertTo-Json -Compress | Write-Output
                if($attempt -eq 2){throw}
            }finally{
                if($outputStream){$outputStream.Dispose()};if($inputStream){$inputStream.Dispose()};if($response){$response.Dispose()};if($request){$request.Dispose()}
            }
        }
    }
    $items+=[ordered]@{file=$file.rfilename;bytes=[long]$file.size;sha256=$h;provider_SHA256_match=$true}
    Save-Receipt 'downloading_and_verifying'
    [ordered]@{file=$file.rfilename;state='size_and_SHA256_PASS';completed=$items.Count;total=15} | ConvertTo-Json -Compress | Write-Output
 }
 Save-Receipt 'all_15_weights_size_and_SHA256_verified'
 Copy-Item -LiteralPath (Join-Path $destination 'verification.json') -Destination (Join-Path $audit 'weights_verification.json') -Force
}finally{$client.Dispose()}
