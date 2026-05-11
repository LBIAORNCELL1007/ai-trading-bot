# Paper-trading launcher -- starts live_bot.py in a NEW PowerShell window
# that survives this Claude session.  Logs go to live_bot.log (append).
#
# Usage:
#     .\start_paper_bot.ps1           # launches with default 4-symbol edge set
#     .\start_paper_bot.ps1 -DryRun   # prints what would run, doesn't start
#
# To stop the bot later: just close the new PowerShell window, or Ctrl+C in it.

param(
    [switch]$DryRun
)

$RepoDir = 'D:\ai-trading-bot'
$Worker  = Join-Path $RepoDir '_paper_bot_worker.ps1'

if (-not (Test-Path $Worker)) {
    Write-Host "ERROR: worker script not found at $Worker" -ForegroundColor Red
    exit 1
}

if ($DryRun) {
    Write-Host "DRY RUN -- would launch new PowerShell window running:"
    Write-Host "  powershell -NoExit -ExecutionPolicy Bypass -File $Worker"
    exit 0
}

Write-Host 'Launching paper bot in a new window...'
Start-Process powershell -ArgumentList @(
    '-NoExit',
    '-ExecutionPolicy', 'Bypass',
    '-File', $Worker
)
Write-Host 'Done. The new window will keep the bot alive (auto-restart on crash).'
Write-Host "Tail logs: Get-Content 'D:\ai-trading-bot\live_bot.log' -Wait -Tail 30"
Write-Host 'Summary:   python paper_summary.py'
