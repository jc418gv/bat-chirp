param(
    [string]$Night,
    [string]$HostName = "vid70",
    [string]$KeyPath = "C:\Users\Jeff\.ssh\jced25519.ppk",
    [string]$RemoteNightRunsRoot = "/media/bag/audio/AudioMoth/work/night-runs",
    [int]$Count = 6,
    [string]$SampleToken,
    [switch]$Fetch,
    [string]$LocalOutputDir = "local/remote-inspection"
)

$plink = "C:/Program Files/PuTTY/plink.exe"
$pscp = "C:/Program Files/PuTTY/pscp.exe"

if (-not (Test-Path $plink)) {
    throw "plink.exe not found at $plink"
}

if (-not (Test-Path $pscp)) {
    throw "pscp.exe not found at $pscp"
}

if (-not (Test-Path $KeyPath)) {
    throw "PuTTY key not found at $KeyPath"
}

if ($Count -le 0) {
    throw "Count must be greater than zero."
}

$resolvedLocalOutputDir = Resolve-Path -LiteralPath "." | ForEach-Object {
    Join-Path $_ $LocalOutputDir
}
New-Item -ItemType Directory -Path $resolvedLocalOutputDir -Force | Out-Null

$remoteBase = if ([string]::IsNullOrWhiteSpace($Night)) {
    $RemoteNightRunsRoot
} else {
    if ($Night -notmatch '^\d{8}$') {
        throw "Night must be an AudioMoth date token like 20260518."
    }
    "$RemoteNightRunsRoot/$Night"
}

$escapedBase = $remoteBase.Replace("'", "'\''")

if ([string]::IsNullOrWhiteSpace($SampleToken)) {
    $remoteCommand = @"
find '$escapedBase' -maxdepth 3 -type f \( -name 'spectrogram_*.png' -o -name 'detections_*.json' \) -printf '%T@ %p\n' | sort -rn | head -n $Count
"@
} else {
    if ($SampleToken -notmatch '^\d{6}$') {
        throw "SampleToken must look like 004245."
    }

    $remoteCommand = @"
find '$escapedBase' -maxdepth 3 -type f \( -name '*_$SampleToken.png' -o -name '*_$SampleToken.json' \) -printf '%T@ %p\n' | sort -rn
"@
}

$rawListing = & $plink -batch -i $KeyPath $HostName $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "Remote listing failed with exit code $LASTEXITCODE."
}

$entries = @()
foreach ($line in $rawListing) {
    if ([string]::IsNullOrWhiteSpace($line)) {
        continue
    }

    $parts = $line -split ' ', 2
    if ($parts.Count -lt 2) {
        continue
    }

    $entries += [pscustomobject]@{
        Epoch = [double]$parts[0]
        Path = $parts[1].Trim()
    }
}

if ($entries.Count -eq 0) {
    Write-Output "No matching artifacts found under $remoteBase"
    exit 0
}

foreach ($entry in ($entries | Sort-Object Epoch -Descending)) {
    $modifiedUtc = [DateTimeOffset]::FromUnixTimeSeconds([long][math]::Floor($entry.Epoch)).UtcDateTime.ToString('u')
    Write-Output "$modifiedUtc`t$($entry.Path)"
}

if (-not $Fetch) {
    exit 0
}

$targetEntries = $entries
foreach ($entry in $targetEntries) {
    $leafName = Split-Path -Path $entry.Path -Leaf
    $destinationPath = Join-Path $resolvedLocalOutputDir $leafName
    & $pscp -batch -i $KeyPath "${HostName}:$($entry.Path)" $destinationPath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to fetch $($entry.Path)"
    }
    Write-Output "Fetched $leafName -> $destinationPath"
}