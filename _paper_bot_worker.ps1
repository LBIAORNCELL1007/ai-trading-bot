# Worker script: runs the paper bot in an auto-restart loop.
# Invoked by start_paper_bot.ps1 inside a new PowerShell window.
# Do NOT run this directly unless you want the loop in your current shell.

$env:PYTHONIOENCODING = 'utf-8'
Set-Location 'D:\ai-trading-bot'

$Python  = 'C:\Python314\python.exe'
$Symbols = 'SOLUSDT,DOGEUSDT,XRPUSDT,AVAXUSDT,ETHUSDT'
$LogFile = 'D:\ai-trading-bot\live_bot.log'

Write-Host '--------------------------------------------------------------'
Write-Host "Paper bot starting at $(Get-Date)" -ForegroundColor Cyan
Write-Host "Symbols: $Symbols"
Write-Host "Log:     $LogFile"
Write-Host '--------------------------------------------------------------'

while ($true) {
    try {
        # -u: unbuffered stdout/stderr so Tee-Object writes the log live.
        # Out-File -Append with explicit flush via redirection.
        & $Python -u live_bot.py --symbols $Symbols --paper --mainnet 2>&1 |
            Tee-Object -FilePath $LogFile -Append
    } catch {
        Write-Host "[wrapper] bot exited with error: $_" -ForegroundColor Red
    }
    $msg = "[wrapper] bot exited at $(Get-Date); restarting in 30s..."
    Write-Host $msg -ForegroundColor Yellow
    Add-Content -Path $LogFile -Value $msg
    Start-Sleep -Seconds 30
}
