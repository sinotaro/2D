# PSD Face Tracker

PSD で描いたキャラクター（`vtuber_layers.psd`）を、Web カメラに映った自分の顔の動きに合わせて動かすブラウザアプリです。
インストール不要で、カメラ映像はブラウザ内だけで処理されます（どこにも送信されません）。

## いちばん簡単な使い方（1ファイル版）

[`vtuber.html`](vtuber.html) をダウンロードして、ダブルクリックで開くだけです（Chrome 推奨）。
「カメラ開始」を押してカメラを許可してください。顔認識の部品をネットから読み込むので、インターネット接続が必要です。

コードを変えたら `python3 tools/build_single.py` で作り直せます。

## 使い方（開発用）

ブラウザのカメラは `https://` か `localhost` でしか使えないので、ローカルサーバー経由で開きます。

```sh
cd 2D
python3 -m http.server 8000
```

ブラウザで <http://localhost:8000/> を開いて **「カメラ開始」** を押し、カメラの使用を許可してください。

- 最初の 0.5 秒ほどで「正面の顔」を基準として記録します。ずれていると感じたら正面を向いて **「正面リセット」**（`C` キー）
- **鏡のように動かす**：オンだと自分が右を向くとキャラも画面の右を向きます
- **なめらかさ**：上げるとブレが減り、下げると反応が速くなります
- **背景**：OBS などで使う場合は「グリーンバック」か、ブラウザソースなら「透過」
- `H` キーで操作パネルを隠せます
- カメラを使わないときは、マウスで顔の向き、クリックで口パクを試せます

### OBS で配信に使う

1. 上の手順でサーバーを起動し、ブラウザで開いてカメラ開始
2. OBS の「ウィンドウキャプチャ」でブラウザを取り込み、背景をグリーンバックにして「クロマキー」フィルタを掛ける

## 動く部分

| カメラで検出するもの | キャラの動き |
| --- | --- |
| 顔の向き（左右・上下）・傾き | 頭全体を立体的に回転（パーツごとの奥行き差で立体感を出す） |
| 顔の位置 | 頭と体が少し移動 |
| まばたき（左右別） | まぶたが閉じる |
| 視線 | 黒目が動く（白目からはみ出さない） |
| 口の開き・笑顔 | 口が開く・口角が上がる |
| 眉の上げ下げ | 眉が動く |
| — | 髪が頭の動きに遅れて揺れる、呼吸、顔を見失ったときの自動まばたき |

顔認識には Google の [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) を CDN から読み込んで使います（初回のみネット接続が必要）。

## ファイル構成

```
index.html, style.css     画面
src/main.js               UI・メインループ・平滑化
src/tracker.js            カメラ → 顔のパラメータ（向き・まばたき・口など）
src/rig.js                パラメータ → 各レイヤーのメッシュ変形
src/renderer.js           WebGL メッシュ描画
assets/model/             PSD から書き出したレイヤー PNG と model.json
tools/extract_psd.py      PSD → assets/model の書き出しスクリプト
```

## PSD を差し替える

1 枚絵からレイヤー分けした PSD を作るには [See-through](https://github.com/shitagaki-lab/see-through) が使えます。
Windows + AMD Radeon（RX 6600 XT など）でローカル実行する手順は [`tools/see-through-win/`](tools/see-through-win/README.md) を参照してください。

```sh
pip install psd-tools numpy
python3 tools/extract_psd.py 新しいキャラ.psd assets/model
```

レイヤー名で役割を判断します。同じ名前のレイヤーを用意すると目・口などが自動で動きます：

`back hair` / `neck` / `topwear` / `neckwear` / `ears` / `face` / `nose` / `mouth` / `eyewhite` / `irides` / `eyelash` / `eyebrow` / `front hair`

- `face` の下にまぶたを閉じたときの肌が描かれている必要があります（目のパーツを消しても穴が開かないこと）
- `eyewhite`・`irides`・`eyelash`・`eyebrow` は左右の目を 1 枚に入れたままで OK（顔の中心で自動的に左右に分けます）
- 知らない名前のレイヤーは、位置に応じて頭か体に付いて動きます
- 動きの大きさは `src/rig.js` の `ROLES`（奥行き `depth`、髪の揺れ `swing`）で調整できます
