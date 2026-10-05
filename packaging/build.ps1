# Builds dist\memeify\memeify.exe, then (if Inno Setup is installed) dist\memeify-setup.exe.
# Run from the repo root:  powershell -File packaging\build.ps1
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
& $py -m PyInstaller packaging\memeify.spec --noconfirm --distpath dist --workpath build
if ($LASTEXITCODE) { throw "PyInstaller failed" }
$out = "dist\memeify"
Copy-Item config.yaml, LICENSE, README.md $out -Force
Copy-Item assets $out -Recurse -Force
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) { $iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1 }
if ($iscc) { & $iscc packaging\memeify.iss } else { Write-Host "Inno Setup not found; skipped the installer. Get it from https://jrsoftware.org/isinfo.php" }
