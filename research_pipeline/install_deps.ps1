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
Write-Host "If you need GPU support, install NVIDIA drivers, CUDA and cuDNN manually as described in TENSORFLOW_GPU_CONFIG.md"