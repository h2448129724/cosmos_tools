param()

# 从脚本自身位置推导仓库根目录（脚本位于 scripts/ 下）
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'Cosmos 工具箱.lnk'
$targetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$launcherScript = Join-Path $RepoRoot '启动Cosmos工具箱.ps1'
$arguments = "-ExecutionPolicy Bypass -File `"$launcherScript`""
$iconPath = "$env:SystemRoot\System32\shell32.dll,220"

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.Arguments = $arguments
$shortcut.WorkingDirectory = $RepoRoot
$shortcut.IconLocation = $iconPath
$shortcut.Description = 'Launch the Cosmos toolbox'
$shortcut.Save()

Write-Host "已创建快捷方式: $shortcutPath"
