# See-through V3 で 1 枚の絵（またはフォルダ内の全画像）をレイヤー分けした PSD にする。
#
#   powershell -ExecutionPolicy Bypass -File run.ps1 -Image C:\path\to\character.png
#   powershell -ExecutionPolicy Bypass -File run.ps1 -Image C:\pics -Resolution 1024
#
# -Mode
#   blockswap … 既定。UNet を少しずつ GPU に載せ替えて 8GB でも動かす（RX 6600 XT 向け）
#   offload   … 公式の --group_offload（約 10GB 必要）
#   standard  … オフロードなし（12〜16GB 必要）
param(
    [Parameter(Mandatory = $true)][string]$Image,
    [ValidateSet("blockswap", "offload", "standard")][string]$Mode = "blockswap",
    [int]$Resolution = 1280,
    [string]$OutDir,
    [switch]$TblrSplit,     # 目・手などを左右別レイヤーに分ける
    [string]$InstallDir = (Join-Path $env:USERPROFILE "see-through")
)

$ErrorActionPreference = "Stop"
$Repo = Join-Path $InstallDir "repo"
$Py = Join-Path $InstallDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "環境が見つかりません。先に setup.ps1 を実行してください（-InstallDir $InstallDir）。" }

$Image = (Resolve-Path $Image).Path
if (-not $OutDir) { $OutDir = Join-Path $InstallDir "output" }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

# MIOpen の畳み込みカーネル探索を短くする（初回の待ち時間対策）
$env:MIOPEN_FIND_MODE = "FAST"
# モデル（合計十数 GB）のダウンロード先。変えたいときは事前に HF_HOME を設定しておく
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $InstallDir "hf-cache" }
$env:PYTHONUTF8 = "1"
# Hugging Face ライブラリの利用統計（バージョン情報など）を送らない
$env:HF_HUB_DISABLE_TELEMETRY = "1"

$script = if ($Mode -eq "blockswap") { "inference\scripts\inference_psd_blockswap.py" } else { "inference\scripts\inference_psd.py" }
$argv = @((Join-Path $PSScriptRoot "launch.py"), $script,
    "--srcp", $Image, "--save_dir", $OutDir, "--resolution", $Resolution, "--save_to_psd")
if ($Mode -eq "offload") { $argv += "--group_offload" }
if ($TblrSplit) { $argv += "--tblr_split" }

if ($Mode -eq "blockswap" -and (Test-Path $Image -PathType Container)) {
    # blockswap 版は 1 枚ずつしか受け付けないので、ここで回す
    $files = Get-ChildItem $Image -File | Where-Object { $_.Extension -match '^\.(png|jpe?g|webp)$' }
    if (-not $files) { throw "$Image に画像（png/jpg/webp）がありません" }
} else {
    $files = @($null)
}

Push-Location $Repo
try {
    foreach ($f in $files) {
        $a = $argv.Clone()
        if ($f) { $a[3] = $f.FullName; Write-Host "`n>>> $($f.Name)" -ForegroundColor Cyan }
        & $Py @a
        if ($LASTEXITCODE -ne 0) { throw "推論に失敗しました（終了コード $LASTEXITCODE）" }
    }
} finally {
    Pop-Location
}
Write-Host "`n完了: $OutDir に <画像名>.psd と <画像名>_depth.psd ができています" -ForegroundColor Green
