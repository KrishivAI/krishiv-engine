# SPDX-License-Identifier: Apache-2.0
#
# Installs the `krishiv` command line binary on Windows (x86-64):
#
#   irm https://krishiv.ai/install.ps1 | iex
#
# Downloads the binary and SHA256SUMS from the GitHub Release, checks the checksum and only then installs it.
# The Windows build ships with the Windows port (#1002); until a release has `krishiv-windows-x86_64.exe` this
# script stops with a clear message. No admin rights, no PATH edits: it prints the line to add.
#
# Environment: KRISHIV_VERSION (default: the latest release), KRISHIV_INSTALL_DIR (default: $HOME\.krishiv\bin),
# KRISHIV_BASE_URL (mirrors and tests), KRISHIV_DRY_RUN=1 (say what would happen, change nothing).
$ErrorActionPreference = 'Stop'

$Repo = 'KrishivAI/krishiv-engine'
$Artifact = 'krishiv-windows-x86_64.exe'

if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw "krishiv-install: no build for $($env:PROCESSOR_ARCHITECTURE)" }

if ($env:KRISHIV_BASE_URL) { $Base = $env:KRISHIV_BASE_URL }
elseif ($env:KRISHIV_VERSION) { $Base = "https://github.com/$Repo/releases/download/v$($env:KRISHIV_VERSION.TrimStart('v'))" }
else { $Base = "https://github.com/$Repo/releases/latest/download" }

$Dir = if ($env:KRISHIV_INSTALL_DIR) { $env:KRISHIV_INSTALL_DIR } else { Join-Path $HOME '.krishiv\bin' }

if ($env:KRISHIV_DRY_RUN -eq '1') {
    Write-Host "krishiv-install: dry run: would download $Base/$Artifact and $Base/SHA256SUMS, verify the checksum, and install $Dir\krishiv.exe"
    return
}

$Tmp = Join-Path ([IO.Path]::GetTempPath()) ("krishiv-install-" + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $Tmp | Out-Null
try {
    Write-Host "krishiv-install: downloading $Artifact"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$Base/$Artifact" -OutFile (Join-Path $Tmp $Artifact)
        Invoke-WebRequest -UseBasicParsing -Uri "$Base/SHA256SUMS" -OutFile (Join-Path $Tmp 'SHA256SUMS')
    } catch {
        throw "krishiv-install: could not download from $Base (does this release have a Windows build?): $_"
    }

    $want = $null
    foreach ($line in Get-Content (Join-Path $Tmp 'SHA256SUMS')) {
        $parts = $line -split '\s+', 2
        if ($parts.Count -eq 2 -and $parts[1].TrimStart('*') -eq $Artifact) { $want = $parts[0]; break }
    }
    if (-not $want) { throw "krishiv-install: SHA256SUMS has no entry for $Artifact" }
    $got = (Get-FileHash -Algorithm SHA256 (Join-Path $Tmp $Artifact)).Hash.ToLower()
    if ($want.ToLower() -ne $got) { throw "krishiv-install: checksum mismatch for $Artifact (expected $want, got $got); nothing installed" }
    Write-Host 'krishiv-install: checksum ok'

    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    Copy-Item -Force (Join-Path $Tmp $Artifact) (Join-Path $Dir 'krishiv.exe')
    Write-Host "krishiv-install: installed $Dir\krishiv.exe ($(& (Join-Path $Dir 'krishiv.exe') --version))"
    if (($env:PATH -split ';') -notcontains $Dir) { Write-Host "krishiv-install: add it to your PATH:  `$env:PATH = `"$Dir;`$env:PATH`"" }
} finally {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}
