param([string]$ResultDirectory = 'output/stereo_crop_bc_test')
$ErrorActionPreference = 'Stop'
$culture = [System.Globalization.CultureInfo]::InvariantCulture
function Number([string]$value) { [double]::Parse($value, $culture) }
function Average($values) {
    if (@($values).Count -eq 0) { return $null }
    (@($values) | Measure-Object -Average).Average
}
function Maximum($values) {
    if (@($values).Count -eq 0) { return $null }
    (@($values) | Measure-Object -Maximum).Maximum
}
function Median($values) {
    $ordered = @($values | Sort-Object)
    if ($ordered.Count -eq 0) { return $null }
    if ($ordered.Count % 2) { return $ordered[[int][math]::Floor($ordered.Count / 2)] }
    ($ordered[$ordered.Count / 2 - 1] + $ordered[$ordered.Count / 2]) / 2
}
$manifest = Get-Content (Join-Path $ResultDirectory 'manifest_snapshot.json') -Raw | ConvertFrom-Json
$frameRows = [System.Collections.Generic.List[object]]::new()
$pointRows = [System.Collections.Generic.List[object]]::new()
$summaries = [System.Collections.Generic.List[object]]::new()
foreach ($variant in @('default_1p5','tolerance_3','full_3','roi_3_perf')) {
    $directory = Join-Path $ResultDirectory $variant
    $log = Get-Content (Join-Path $directory 'tracking_log.txt')
    $statuses = @{}
    $keypoints = @{}
    $currentFrame = 0
    foreach ($line in $log) {
        if ($line -match '===== 第 ([0-9]+) 帧') { $currentFrame = [int]$Matches[1] }
        if ($line -match '特征点: ([0-9]+)') { $keypoints[$currentFrame] = [int]$Matches[1] }
        if ($line -match '^[[]StereoFrontend[]] frame=([0-9]+).* pairs=([0-9]+) match_ms=([0-9.eE+-]+)') {
            $index = [int]$Matches[1]
            $statuses[$index] = [pscustomobject]@{Status='FAIL';Count=[int]$Matches[2];MatchMs=(Number $Matches[3]);Width=0;Height=0;Retry=0;Reason=''}
            if ($line -match 'match_ok=1') { $statuses[$index].Status = 'OK' }
            if ($line -match 'roi=([0-9]+)x([0-9]+) full_retry=([01])') {
                $statuses[$index].Width = [int]$Matches[1]
                $statuses[$index].Height = [int]$Matches[2]
                $statuses[$index].Retry = [int]$Matches[3]
            }
            if ($line -match 'reason=([^ ]+)') { $statuses[$index].Reason = $Matches[1] }
        }
    }
    for ($i=1; $i -le $manifest.frames.Count; $i++) {
        $truth = $manifest.frames[$i-1]
        if (!$statuses.ContainsKey($i)) { throw "Missing frame status: $variant frame $i" }
        $status = $statuses[$i]
        $points = @(Import-Csv (Join-Path $directory "disparity_f$i.csv"))
        if ($points.Count -ne $status.Count) { throw "Point count mismatch: $variant frame $i" }
        $errors = @(); $verticalErrors = @(); $disparities = @(); $shifts = @()
        foreach ($p in $points) {
            $d = Number $p.disparity_px
            $dy = (Number $p.right_y) - (Number $p.left_y)
            $e = $d - [double]$truth.disparity
            $ey = $dy - [double]$truth.dy_px
            $errors += [math]::Abs($e); $verticalErrors += [math]::Abs($ey)
            $disparities += $d; $shifts += $dy
            $pointRows.Add([pscustomobject]@{variant=$variant;frame=$i;frame_status=$status.Status;left_index=[int]$p.left_index;left_x=(Number $p.left_x);left_y=(Number $p.left_y);right_x=(Number $p.right_x);right_y=(Number $p.right_y);disparity_px=$d;truth_disparity_px=$truth.disparity;disparity_error_px=$e;vertical_shift_px=$dy;truth_vertical_shift_px=$truth.dy_px;vertical_error_px=$ey})
        }
        $frameRows.Add([pscustomobject]@{variant=$variant;frame=$i;source=$truth.src;status=$status.Status;bc_keypoints=$keypoints[$i];pairs=$points.Count;left_indices=($points.left_index -join '|');truth_disparity_px=$truth.disparity;median_disparity_px=(Median $disparities);disparity_mae_px=(Average $errors);max_disparity_error_px=(Maximum $errors);mean_vertical_shift_px=(Average $shifts);vertical_mae_px=(Average $verticalErrors);max_vertical_error_px=(Maximum $verticalErrors);match_ms=$status.MatchMs;roi_width=$status.Width;roi_height=$status.Height;full_retry=$status.Retry;failure_reason=$status.Reason})
    }
    $frames = @($frameRows | Where-Object variant -eq $variant)
    $points = @($pointRows | Where-Object variant -eq $variant)
    $dispErrors = @($points | ForEach-Object { [math]::Abs($_.disparity_error_px) })
    $dyErrors = @($points | ForEach-Object { [math]::Abs($_.vertical_error_px) })
    $squared = @($points | ForEach-Object { $_.disparity_error_px * $_.disparity_error_px })
    $rmse = if ($squared.Count) { [math]::Sqrt((Average $squared)) } else { $null }
    $summaries.Add([pscustomobject]@{variant=$variant;frames=$frames.Count;successful_frames=@($frames | Where-Object status -eq OK).Count;total_pairs=$points.Count;mean_pairs=(Average $frames.pairs);mean_match_ms=(Average $frames.match_ms);median_match_ms=(Median $frames.match_ms);full_retry_count=($frames.full_retry | Measure-Object -Sum).Sum;disparity_mae_px=(Average $dispErrors);disparity_rmse_px=$rmse;max_disparity_error_px=(Maximum $dispErrors);mean_vertical_shift_px=(Average $points.vertical_shift_px);vertical_mae_px=(Average $dyErrors);max_vertical_error_px=(Maximum $dyErrors)})
    $successfulPoints = @($points | Where-Object frame_status -eq OK)
    $successfulErrors = @($successfulPoints | ForEach-Object { [math]::Abs($_.disparity_error_px) })
    $summaries[$summaries.Count-1] | Add-Member NoteProperty successful_frame_pairs $successfulPoints.Count
    $summaries[$summaries.Count-1] | Add-Member NoteProperty successful_frame_disparity_mae_px (Average $successfulErrors)
    $summaries[$summaries.Count-1] | Add-Member NoteProperty successful_frame_max_disparity_error_px (Maximum $successfulErrors)
}
$frameRows | Export-Csv (Join-Path $ResultDirectory 'frames.csv') -NoTypeInformation -Encoding UTF8
$pointRows | Export-Csv (Join-Path $ResultDirectory 'points.csv') -NoTypeInformation -Encoding UTF8
$comparison = @()
foreach ($i in 1..$manifest.frames.Count) {
    $a = @($pointRows | Where-Object { $_.variant -eq 'full_3' -and $_.frame -eq $i })
    $b = @($pointRows | Where-Object { $_.variant -eq 'roi_3_perf' -and $_.frame -eq $i })
    $sameIds = ($a.left_index -join '|') -eq ($b.left_index -join '|')
    $delta = @()
    if ($sameIds) {
        for ($j=0; $j -lt $a.Count; $j++) {
            $dx = $a[$j].right_x-$b[$j].right_x; $dy = $a[$j].right_y-$b[$j].right_y
            $delta += [math]::Sqrt($dx*$dx+$dy*$dy)
        }
    }
    $comparison += [pscustomobject]@{frame=$i;full_pairs=$a.Count;roi_pairs=$b.Count;same_indices=$sameIds;max_right_point_delta_px=(Maximum $delta)}
}
$comparison | Export-Csv (Join-Path $ResultDirectory 'full_vs_roi.csv') -NoTypeInformation -Encoding UTF8
$result = [pscustomobject]@{summaries=$summaries;full_roi_identical_indices=@($comparison | Where-Object same_indices -eq $true).Count;full_roi_max_point_delta_px=(Maximum $comparison.max_right_point_delta_px)}
$result | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $ResultDirectory 'summary.json') -Encoding UTF8
$result | ConvertTo-Json -Depth 10
