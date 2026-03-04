# PowerShell script to create a Desktop shortcut for IDS ML Analyzer
# Run this script to generate a link on your Desktop with a custom icon.

$S_Name = "IDS ML Analyzer"
$S_Icon = Join-Path $PSScriptRoot "app_icon.ico"
$S_Target = Join-Path $PSScriptRoot "run_app.bat"
$S_WorkDir = $PSScriptRoot

if (-not (Test-Path $S_Icon)) {
    # Fallback to PNG if ICO doesn't exist (though it should now)
    $S_Icon = Join-Path $PSScriptRoot "app_icon.png"
}

$WshShell = New-Object -ComObject WScript.Shell
$ShortcutPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "$S_Name.lnk"

# Remove existing shortcut if it exists to refresh the icon
if (Test-Path $ShortcutPath) {
    Remove-Item $ShortcutPath -Force
}

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $S_Target
$Shortcut.WorkingDirectory = $S_WorkDir
$Shortcut.IconLocation = "$S_Icon,0" # This correctly sets the .ico in Windows
$Shortcut.Save()

Write-Host "[OK] Desktop shortcut created: $ShortcutPath" -ForegroundColor Cyan
Write-Host "[OK] Custom icon applied: $S_Icon" -ForegroundColor Green
