param([switch]$Install)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
# Avoid installing thousands of package files on the Google Drive filesystem.
$taskEnvironment = Join-Path ([System.IO.Path]::GetTempPath()) 'pdf-insight-venv-311'
$taskPython = Join-Path $taskEnvironment 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    py -3.11 -m venv $taskEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create the Python 3.11 environment.' }
    $Install = $true
}
if ($Install) {
    & $taskPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
& $taskPython server.py
