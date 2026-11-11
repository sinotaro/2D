# 競馬 着順予想ツール (JV-Link + LightGBM)

JRA-VAN Data Lab. の **JV-Link** から過去レースを取得し、**発走前に分かる情報だけ**を使って
各レースの着順を予想する機械学習ツールです。

```
JV-Link ──(Windows/32bit)──▶ data/raw/*.jv ──load──▶ SQLite ──features──▶ LightGBM ──▶ 予想
          keiba.jvlink_client                 keiba.db        keiba.features   keiba.model   keiba.predict
```

## 出力例

```
■ 2025-07-19 京都11R 架空ステークス11  (2025071908010111)
 予想  馬番  馬名                  騎手        勝率   複勝率
 ◎ 1     5  ソムクムワ            騎手5        39.8%   79.8%
 ○ 2     8  トロチルミコ          騎手36       24.6%   54.0%
 ▲ 3     1  ヌアオマフヌリ        騎手42       18.7%   59.3%
```

* **予想**: LambdaRank モデルのスコア順 (= 予想着順)
* **勝率**: 1着確率 (レース内で合計 100%)
* **複勝率**: 3着以内確率 (レース内で合計 300%)

## セットアップ

```bash
pip install -r requirements.txt
```

### JV-Link について (Windows)

* JRA-VAN Data Lab. の会員登録と JV-Link のインストールが必要です。
* JV-Link は **32bit COM** なので、取得だけは **32bit 版 Python + pywin32** で実行します。
  `keiba/jvlink_client.py` は標準ライブラリと pywin32 しか使わないので、32bit 側に
  pandas や LightGBM を入れる必要はありません (LightGBM は 32bit 版 Windows 用ホイールがありません)。
* 学習・予想は 64bit Python (Windows / Mac / Linux どれでも可) で行います。
  `data/` フォルダを共有すれば 32bit と 64bit で分担できます。

```bat
py -3-32 -m pip install pywin32
```

## 使い方

### 1. 過去データの取得 (初回のみ・時間がかかります)

```bat
py -3-32 -m keiba.jvlink_client setup --from 20150101
```

`data/raw/` に RA (レース詳細)・SE (馬毎レース情報)・HR (払戻) レコードが保存されます。

### 2. DB に取り込んで学習

```bash
python -m keiba load
python -m keiba train                       # 期間の後ろ 20% を検証に使う
python -m keiba train --valid-from 2025-01-01
```

検証期間での成績 (的中率・NDCG・単勝回収率など) と重要な特徴量が表示され、
最後に全期間で学習し直したモデルが `models/` に保存されます。

### 3. 週末の予想

```bat
:: 木曜〜金曜: 今週の出馬表を取得
py -3-32 -m keiba.jvlink_client week
:: 当日 (発走 70 分前頃〜): 馬体重を取得するとさらに良い
py -3-32 -m keiba.jvlink_client weights --date 20261115
```

```bash
python -m keiba load
python -m keiba predict --date 2026-11-15
python -m keiba predict --date 2026-11-15 --jyo 05 --top 5 --out out/20261115.csv
```

### 4. 差分更新 (毎週)

```bat
py -3-32 -m keiba.jvlink_client update
```

```bash
python -m keiba load && python -m keiba train
```

### JV-Link が無くても試す

JV-Data と同じ固定長フォーマットの架空データを生成して、取込〜学習〜予想を一通り実行します。

```bash
python -m keiba demo
```

## 特徴量 (すべて発走前に分かるもの)

| 区分 | 内容 |
|---|---|
| 当該レース | 枠番・馬番・頭数・馬齢・性別・斤量 (レース平均との差)・馬体重/増減・距離・芝ダート・競馬場・馬場状態・天候・グレード・クラス・ハンデ戦か |
| 馬の過去成績 | 通算勝率/複勝率、直近 5 走の着順率・勝ち馬とのタイム差・上がり 3F (レース内の差)・4 角位置 (脚質)・クラス、前走からの間隔/距離変更/斤量変更/乗り替わり/芝ダ替わり |
| 条件別適性 | 同じ芝ダ・同じ距離帯・同じコース・道悪・同じ騎手での過去の出走数と複勝率 |
| 騎手・調教師 | 通算と直近 365 日の勝率/複勝率 (少数サンプルは平均方向へ平滑化) |
| レース内相対値 | 主要な指標について、同じレースの他馬の中での順位と平均との差 |

### リーク (未来の情報の混入) 対策

* 馬の成績は「そのレースより前の出走」だけから計算 (グループ内で 1 行ずらす)。
* 騎手・調教師の成績は**日単位**で集計し前日までの値を使うので、同じ日の先のレースの結果は混ざりません。
* **単勝オッズ・人気は特徴量に使っていません** (確定オッズは発走直前まで分からないため)。評価時の回収率計算にだけ使います。
* `tests/test_features.py::test_no_future_leakage` で、ある日以降の結果を書き換えても
  その日以前の特徴量が 1 ビットも変わらないことを検証しています。

注意: 馬場状態・天候は成績データ (確定版) の値を使っているので、発表前に予想する場合は
実際と異なることがあります。

## モデル

| モデル | 目的 | 出力 |
|---|---|---|
| `ranker` | LightGBM LambdaRank (1着=3, 2着=2, 3着=1, 他=0) | 予想着順 |
| `win` | 二値分類 (1着か) | 勝率 (レース内で正規化) |
| `top3` | 二値分類 (3着以内か) | 複勝率 (レース内で正規化) |

時系列で学習/検証期間を分けて (シャッフルしない) 早期終了で木の本数を決め、
最終モデルは全期間で学習し直します。

評価指標: `top1_win_rate` (◎の勝率)、`top1_top3_rate` (◎の複勝率)、`ndcg@1/@3`、
着順の順位相関、`top1_tansho_return` (◎の単勝を買い続けた場合の回収率)、
`ev_tansho_return` (勝率 × オッズ > 1.2 の馬だけ買った場合の回収率)。

## 構成

```
keiba/
  jvlink_client.py   JV-Link から取得 (Windows, 32bit Python)
  jvdata/spec.py     JV-Data レコードのバイト位置定義
  jvdata/parser.py   固定長レコードのパース
  db.py              SQLite への蓄積 (出馬表→速報→確定 を新しい方で上書き)
  features.py        特徴量生成
  model.py           学習・評価・推論
  predict.py         予想の表示
  synthetic.py       動作確認用の架空データ生成
tests/               pytest
```

## 補足・今後の拡張

* レコードのバイト位置は JV-Data 仕様書 Ver.4.x に準拠しています。仕様書の改版で位置が変わった場合は
  `keiba/jvdata/spec.py` のみ修正してください。実データ取込時にレコード長が合わない行は警告してスキップします。
* 血統 (UM/HN レコードの父・母父) や調教 (HC/WC: 坂路・ウッド調教) を加えると精度向上が期待できます。
* 発走直前のオッズ (速報 0B31) を特徴量に加えるオプションも追加可能です (的中率は上がりますが回収率は上がりにくい)。
* 馬券の購入判断は自己責任でお願いします。
