# Зібрати інсталятор для Pi 5 (aarch64). Потбен інтернет для wheels.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
py -3 deploy\build_pi_package.py @args
