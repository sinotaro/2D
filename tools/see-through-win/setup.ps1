# See-through V3 を Windows + AMD Radeon (ROCm / TheRock) で動かす環境を作る。
#
# 使い方（PowerShell）:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -InstallDir D:\see-through -Gfx gfx1032
#
# 何度実行しても大丈夫（既にあるものは使い回す）。
param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE "see-through"),
    [string]$Gfx = "gfx1032",            # RX 6600 / 6600 XT
    [string]$PythonVersion = "3.12",
    [ValidateSet("auto", "stable", "nightly")]
    [string]$Channel = "auto",
    [string]$Commit = "df019de5129d6c4b406587a14c3501669441a783"  # 動作確認した see-through のコミット
)

$ErrorActionPreference = "Stop"
$ToolDir = $PSScriptRoot
$Repo = Join-Path $InstallDir "repo"
$Venv = Join-Path $InstallDir ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Indexes = @{
    stable  = "https://stable.repo.amd.com/rocm/whl-next/"
    nightly = "https://nightly.repo.amd.com/rocm/whl-next/"
}

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "失敗しました: $exe $($argv -join ' ')" }
}

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}
function Test-Python {
    $ErrorActionPreference = "Continue"  # py が stderr に出すメッセージで止まらないように
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) { return $false }
    & py "-$PythonVersion" -c "import struct, sys; sys.exit(0 if struct.calcsize('P') == 8 else 1)" 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
}
# 足りないツールを winget（Windows 10/11 標準のパッケージ管理）で入れる
function Install-WithWinget($id, $name, $url) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "$name が見つかりません。$url からインストールして、PowerShell を開き直してから再実行してください。"
    }
    Write-Host "$name が無いので winget でインストールします..."
    & winget install --id $id -e --source winget --accept-package-agreements --accept-source-agreements
    Refresh-Path
}

Step "前提ツールの確認"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Install-WithWinget "Git.Git" "Git" "https://git-scm.com/download/win"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git のインストール後も git が見つかりません。ウィンドウを閉じて setup.bat をもう一度実行してください。"
    }
}
if (-not (Test-Python)) {
    Install-WithWinget "Python.Python.$PythonVersion" "Python $PythonVersion" "https://www.python.org/downloads/windows/"
    if (-not (Test-Python)) {
        throw "Python $PythonVersion のインストール後も py -$PythonVersion で起動できません。ウィンドウを閉じて setup.bat をもう一度実行してください。"
    }
}
git config --global core.longpaths true

Step "see-through のソースを取得 ($InstallDir)"
New-Item -ItemType Directory -Force $InstallDir | Out-Null
if (-not (Test-Path (Join-Path $Repo ".git"))) {
    Run git @("clone", "https://github.com/shitagaki-lab/see-through", $Repo)
}
Run git @("-C", $Repo, "fetch", "origin")
Run git @("-C", $Repo, "checkout", "--detach", $Commit)

# Linux 版の `ln -sf common/assets assets` の代わり（ジャンクションなので管理者権限は不要）
$Assets = Join-Path $Repo "assets"
if (-not (Test-Path $Assets)) {
    New-Item -ItemType Junction -Path $Assets -Target (Join-Path $Repo "common\assets") | Out-Null
}

Step "仮想環境を作成"
if (-not (Test-Path $Py)) {
    Run py @("-$PythonVersion", "-m", "venv", $Venv)
}
Run $Py @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

Step "PyTorch (ROCm, $Gfx) をインストール"
# torch の依存パッケージは先に PyPI から入れておき、torch 本体は AMD のインデックスだけから取る
# （--extra-index-url で混ぜると PyPI の CUDA/CPU 版 torch が選ばれることがあるため）
Run $Py @("-m", "pip", "install", "filelock", "typing-extensions", "sympy", "networkx", "jinja2", "fsspec", "numpy==2.2.6")
$torchPkgs = @("torch[device-$Gfx]", "torchvision[device-$Gfx]", "torchaudio")
$channels = if ($Channel -eq "auto") { @("stable", "nightly") } else { @($Channel) }
$ok = $false
foreach ($ch in $channels) {
    Write-Host "-> $ch チャンネル: $($Indexes[$ch])"
    $argv = @("-m", "pip", "install", "--index-url", $Indexes[$ch]) + $torchPkgs
    if ($ch -eq "nightly") { $argv += "--pre" }
    & $Py @argv
    if ($LASTEXITCODE -eq 0) {
        & $Py (Join-Path $ToolDir "check_gpu.py") --quick
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        Write-Warning "$ch 版の torch は入りましたが GPU で動きませんでした。次を試します。"
    }
}
if (-not $ok) {
    throw "ROCm 版 PyTorch を GPU で動かせませんでした。Adrenalin ドライバを最新にして再実行してください（README の「うまくいかないとき」も参照）。"
}

Step "see-through の依存パッケージをインストール"
# 入れた ROCm 版 torch が PyPI 版で上書きされないよう、バージョンを固定してから入れる
$Constraints = Join-Path $InstallDir "constraints-rocm.txt"
& $Py -m pip freeze | Select-String -Pattern '^(torch|torchvision|torchaudio|rocm|amd-torch)' |
    ForEach-Object { $_.Line } | Set-Content -Encoding ascii $Constraints
Get-Content $Constraints | ForEach-Object { Write-Host "  固定: $_" }
Push-Location $Repo
try {
    Run $Py @("-m", "pip", "install", "-r", "requirements.txt", "-c", $Constraints)
} finally {
    Pop-Location
}

Step "動作チェック"
Run $Py @((Join-Path $ToolDir "check_gpu.py"))
Push-Location $Repo
try {
    $ImportLog = Join-Path $InstallDir "import_check.log"
    & $Py (Join-Path $ToolDir "check_imports.py") $ImportLog
    if ($LASTEXITCODE -ne 0) {
        throw "読み込めないモジュールがあります（上の NG の行を参照）。詳細は $ImportLog にあります。"
    }
} finally {
    Pop-Location
}

Write-Host "`n準備完了です。次のように実行してください:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$(Join-Path $ToolDir 'run.ps1')`" -Image C:\path\to\character.png"
if ($InstallDir -ne (Join-Path $env:USERPROFILE "see-through")) {
    Write-Host "  （-InstallDir $InstallDir も付けてください）"
}
