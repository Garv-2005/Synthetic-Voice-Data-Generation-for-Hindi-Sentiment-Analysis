param(
    [string]$venvName = "venv"
)

Write-Host "Creating virtual environment: $venvName"
python -m venv $venvName

$venvPython = Join-Path -Path $venvName -ChildPath "Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
    Write-Host "ERROR: Could not find virtualenv python executable at $venvPython" -ForegroundColor Red
    exit 1
}

Write-Host "Upgrading pip in virtual environment"
Start-Process -FilePath $venvPython -ArgumentList '-m','pip','install','--upgrade','pip' -NoNewWindow -Wait

Write-Host "Installing dependencies from requirements.txt"
Start-Process -FilePath $venvPython -ArgumentList '-m','pip','install','-r','requirements.txt' -NoNewWindow -Wait

Write-Host "Installation complete. To activate the virtual environment run:"
Write-Host "  .\$venvName\Scripts\Activate.ps1"
Write-Host ""
Write-Host "GPU Setup Notes:"
Write-Host "  - For RTX 4500 Ada: Use Python 3.11 + TensorFlow 2.16.1 for best GPU performance"
Write-Host "  - For development: Current setup works on CPU (slower but functional)"
Write-Host "  - See GPU_SETUP_ISSUES_AND_WORKAROUNDS.md for detailed instructions"