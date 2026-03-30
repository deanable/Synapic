<#
PowerShell installer script for Synapic (install.ps1)
- Silent mode via -Silent (alias -s)
- Behavior:
  - Checks for Python in PATH; if missing, downloads and silent-installs Python 3.11.5 64-bit to C:\Python311
  - Clones Synapic from GitHub if repo not present, otherwise pulls latest
  - Creates a Python virtual environment at .venv
  - Installs dependencies from requirements.txt inside the venv
  - Runs unit tests (pytest) from tests/unit
  - Verbose mode prints progress; Silent mode suppresses logs
#>

param(
    [Alias("s")]
    [switch]$Silent
)

function Write-Log {
    param([string]$Message)
    if (-not $Silent) {
        Write-Host "[install] $Message"
    }
}

$Root = (Get-Location).Path

Write-Log "Starting Synapic installer"

# Step 1: Ensure Python is available
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Log "Python not found in PATH. Installing Python 3.11.5 (64-bit) silently."
    $installer = "$env:TEMP\python_installer.exe"
    $pyUrl = "https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe"
    try {
        Invoke-WebRequest -Uri $pyUrl -OutFile $installer -UseBasicParsing -ErrorAction Stop
    } catch {
        Write-Log "Failed to download Python installer from $pyUrl"
        exit 1
    }
    $args = @("/quiet", "InstallAllUsers=1", "PrependPath=1", "TargetDir=C:\\Python311")
    $proc = Start-Process -FilePath $installer -ArgumentList $args -Wait -NoNewWindow -PassThru
    if (-not (Test-Path 'C:\\Python311\\python.exe')) {
        Write-Log "Python installation failed."
        exit 1
    }
    $env:PATH += ";C:\\Python311;C:\\Python311\\Scripts"
} else {
    Write-Log "Python found: $($pythonCmd.Path)"
}

# Step 2: Get source code
if (Test-Path '.git') {
    Write-Log "Updating existing Synapic repo..."
    git fetch --all 2>$null
    git reset --hard origin/main 2>$null
} else {
    Write-Log "Cloning Synapic repository from GitHub..."
    git clone https://github.com/Dean-Kruger/Synapic.git .
}

# Step 3: Setup virtual environment
if (-not (Test-Path '.venv')) {
    Write-Log "Creating Python virtual environment (.venv)"
    & "$env:PYTHON" -m venv .venv 2>$null
}

if (Test-Path '.venv\Scripts\activate.ps1') {
    . .venv\Scripts\activate.ps1
}

$PipExe = ".\\.venv\\Scripts\\pip.exe"
if (-not (Test-Path $PipExe)) {
    Write-Log "pip not found in virtual environment; attempting to use Python to install requirements."
    $PipExe = ".\\.venv\\Scripts\\python.exe" + " -m pip"
}

if (Test-Path 'requirements.txt') {
    Write-Log "Installing dependencies from requirements.txt..."
    & $PipExe install -r requirements.txt 2>$null
}

# Ensure pytest is available for unit tests
if (-not (Get-Command pytest -ErrorAction SilentlyContinue)) {
    Write-Log "Installing pytest into the virtual environment..."
    & $PipExe install pytest 2>$null
}

# Step 4: Run unit tests
if (Test-Path 'tests\\unit') {
    Write-Log "Running unit tests..."
    & ".\\.venv\\Scripts\\pytest.exe" tests\\unit -q
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

Write-Log "Installation complete."
exit 0
