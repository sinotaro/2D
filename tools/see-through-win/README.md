# See-through V3 を Windows + RX 6600 XT で動かす

[See-through](https://github.com/shitagaki-lab/see-through)（1 枚のアニメ絵を髪・顔・目・服など最大 23 パーツの PSD に分解する AI）を、
Windows の AMD Radeon RX 6600 XT（VRAM 8GB, `gfx1032`）でローカル実行するためのスクリプトです。
できた PSD はそのままこのリポジトリの `tools/extract_psd.py` に渡せます。

## 仕組み

| 項目 | 内容 |
| --- | --- |
| GPU ドライバ層 | AMD 公式の [TheRock](https://github.com/ROCm/TheRock) 版 ROCm（Windows ネイティブ、`pip` で入る） |
| PyTorch | `torch[device-gfx1032]`（AMD のインデックス `repo.amd.com/rocm/whl-next` から） |
| 8GB 対策 | 公式の `inference_psd_blockswap.py`（UNet を少しずつ GPU に載せ替える）を既定で使用 |
| attention | Windows 版 ROCm には RX 6000 向けの省メモリ attention がないため、`launch.py` で attention を分割計算に差し替え（結果は同一） |

WSL2 や ZLUDA、DirectML は使いません（RX 6000 シリーズは WSL2 の ROCm 非対応、DirectML は bf16 とモデルの一部処理が動かないため）。

## 必要なもの

- Windows 10 / 11（64bit）
- **AMD Software: Adrenalin Edition の最新ドライバ**（古いと ROCm から GPU が見えません）
- [Python 3.12](https://www.python.org/downloads/windows/) と [Git for Windows](https://git-scm.com/download/win)
  （入っていなければ setup が winget で自動インストールします）
- メインメモリ 16GB 以上（32GB 推奨。blockswap はモデルの大部分を RAM に置きます）
- 空きディスク 約 30GB（Python パッケージ 数 GB ＋ モデル 十数 GB）

## セットアップ

`setup.bat` をダブルクリックするか、PowerShell で:

```powershell
cd 2D\tools\see-through-win
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`%USERPROFILE%\see-through` に次のものが作られます（`-InstallDir D:\see-through` で変更可）。

```
see-through\
  repo\                 see-through 本体（動作確認したコミットに固定）
  .venv\                Python 仮想環境（ROCm 版 PyTorch 入り）
  hf-cache\             初回実行時にダウンロードされるモデル
  output\               結果の PSD
  constraints-rocm.txt  ROCm 版 torch が上書きされないための固定リスト
```

最後に `check_gpu.py` が GPU で計算できるか確かめ、`OK: GPU で計算できます` と出れば完了です。
PyTorch はまず AMD の stable チャンネル、だめなら nightly チャンネルから入れます（`-Channel nightly` で nightly 固定）。

## 実行

画像を `run.bat` にドラッグ＆ドロップするか:

```powershell
powershell -ExecutionPolicy Bypass -File run.ps1 -Image C:\path\to\character.png
```

`%USERPROFILE%\see-through\output\` に `character.psd`（レイヤー分け）と `character_depth.psd`（奥行き）ができます。
初回はモデルのダウンロードがあるので時間がかかります。

| オプション | 説明 |
| --- | --- |
| `-Image <フォルダ>` | フォルダ内の png / jpg / webp をすべて処理 |
| `-Resolution 1024` | VRAM 不足（`out of memory`）になるときや速くしたいとき。既定 1280 |
| `-TblrSplit` | 目・手などを左右別のレイヤーに分ける |
| `-Mode offload` / `standard` | 公式の `--group_offload`（約 10GB）/ オフロードなし（12〜16GB）。6600 XT では使わない |
| `-OutDir <フォルダ>` | 出力先 |

### このリポジトリのアバターに使う

リポジトリの `2D` フォルダで:

```powershell
& "$env:USERPROFILE\see-through\.venv\Scripts\python.exe" tools\extract_psd.py "$env:USERPROFILE\see-through\output\character.psd" assets\model
```

See-through のレイヤー名（`front hair`・`back hair`・`face`・`eyewhite`・`irides` など）は
このアプリの役割名と同じなので、目や口がそのまま動きます（この用途では `-TblrSplit` は付けないでください。左右分割はアプリ側で行います）。

## うまくいかないとき

- **`NG: GPU が見つかりません`**：Adrenalin ドライバを最新にして再起動。`.venv\Scripts\python.exe check_gpu.py` で再確認できます
- **`setup.ps1` で torch のインストールに失敗する**：`-Channel nightly` を付けて再実行
- **`out of memory`**：`-Resolution 1024`、それでもだめなら `$env:SEETHROUGH_SDPA_CHUNK_MB=256` を設定してから実行。
  ブラウザなど VRAM を使う他のアプリも閉じてください
- **初回の実行がとても遅い**：MIOpen が畳み込みカーネルをコンパイル・キャッシュしています。2 回目からは速くなります
- **RAM 不足で落ちる**：Windows の仮想メモリ（ページファイル）を 32GB 以上にしてください
- **ROCm 版でどうしても動かない**：RX 6600 XT は AMD の公式サポート外（コミュニティ扱い）です。
  [TheRock の RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) で gfx1032 の状況を確認するか、
  NVIDIA GPU の PC か [Hugging Face のデモ](https://huggingface.co/spaces/24yearsold/see-through-demo) を使ってください

## ファイル

| ファイル | 役割 |
| --- | --- |
| `setup.ps1` / `setup.bat` | 環境構築 |
| `run.ps1` / `run.bat` | 推論の実行 |
| `launch.py` | attention を分割計算に差し替えてから see-through のスクリプトを実行 |
| `check_gpu.py` | ROCm 版 PyTorch が GPU で正しく計算できるかの確認 |
