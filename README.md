# SOMA Fighter Web

NVIDIA [MotionBricks](https://github.com/NVlabs/motionbricks)（SOMA モデル）を**ブラウザの WebGPU で動かし**、その場で生成した動きだけで戦う 1 対 1 の格闘ゲーム。サーバーは要らず、GitHub Pages の静的ファイルだけで動く。

用意したアニメーションは再生しない。「構え」「ジャブ」「ストレート」「ミドルキック」「のけぞり」「ダウン」のキーポーズだけをコードで作り（`src/poses.js`）、相手の今の位置に合わせて置いて（打撃は IK で手足の先を相手の頭・胴へ届かせて）、そこまでの動きを MotionBricks が毎回生成する。2 人分の生成は 1 回の推論にまとめる。

デスクトップ版 SOMA Fighter（TensorRT サーバー + ブラウザ描画）の考え方を参考にした、最小限のブラウザ版。

## 遊ぶ

GitHub Pages の URL を開き、「モデルを読み込んで始める」を押す。初回は NVIDIA の GitHub からモデル（Mixed FP16 で 380 MB）をダウンロードしてブラウザに保存する（2 回目からは保存したものを使う）。

| 操作 | キー |
|---|---|
| 前後に動く | A / D（矢印も可） |
| 横に動く（奥 / 手前） | W / S |
| ジャブ | J（左クリック） |
| ストレート | L（中クリック） |
| ミドルキック | K（右クリック） |

2 本先取。届かない距離で打つと空振りになる。打たれると打撃は止まる。

### 必要なもの

- WebGPU が使えるデスクトップの Chrome / Edge（WebGPU が無いと WASM（CPU）で動くが、かなり遅い）
- 数 GB の空きメモリ

### 測った値

| 環境 | 1 回の生成（2 人分） |
|---|---|
| WebGPU、NVIDIA Turing（Claude デスクトップの Chromium 152） | 約 190〜270 ms |
| Node の onnxruntime（CPU） | 約 80 ms |

生成にかかる時間のぶん先の姿勢から生成を始め、届いたら続きとしてつなぐ（`src/motion.js`）。間に合わなければ 0.18 秒で混ぜてつなぐ。

## 仕組み

```
src/
  soma.js          骨格（30 関節）、順運動学、MotionBricks の入力エンコード・出力デコード・端点補正
  poses.js         キーポーズ：回転と 2 関節 IK で作る構え・打撃・のけぞり・ダウン
  motion.js        Track（再生中の動きと、生成した動きのつなぎ方）と Scheduler（2 人分を 1 回にまとめる）
  infer.worker.js  ONNX Runtime Web（WebGPU、だめなら WASM）、モデルのダウンロードと Cache Storage への保存
  render.js        three.js：カプセルの人形、床、2 人を横から映すカメラ
  main.js          試合：入力、CPU、当たり判定、ラウンド
assets/soma_rig.json   公式の骨格 USD から取り出したレスト姿勢
tools/
  extract_rig.py   USD → soma_rig.json（uv run --with usd-core python tools/extract_rig.py <usd>）
  smoke.mjs        Node でモデルを動かす試験（エンコード → 推論 → デコードの誤差）
  serve.mjs        ローカル用の静的サーバー
```

- 入力は「直近 4 ポーズ（文脈）+ 目標ポーズ 1 枚」。最後の文脈が +X を向くように整列して、根の位置・向き・速度、関節の位置、6D 回転にする（公式 `kernels.py` と同じ）。
- 出力（24〜64 フレーム）の両端 3 フレームを除き、始まりを文脈の最後に、終わりを目標に合わせる（公式の端点補正と同じ）。だから打撃は IK で置いた点へ正確に届く。
- 打撃は短い生成（6 トークン = 18 フレーム）を 1.6〜2.1 倍速で再生する。当たったかは、手足の先と相手の頭（球）・胴（カプセル）の距離で判定し、ダメージは当たる瞬間の速さで変わる。

## ローカルで動かす

```bash
npm install
npm run serve
```

http://127.0.0.1:8080/ を開く。ダウンロード済みのモデルを `models/` に置けば `?model=models/mmm_v6_soma_fp16.onnx` で使える（`models/` は Git に入れない）。`?ep=wasm` で WASM、`?debug` で `window.game` から状態を読める。

Node で推論を試す：

```bash
npm run smoke -- models/mmm_v6_soma_fp16.onnx
```

## ライセンス

コードは Apache-2.0。MotionBricks のコード（Apache-2.0）を移した部分、骨格データ（CC-BY-4.0）、モデルの重み（NVIDIA Open Model Agreement、このリポジトリには含めない）については [NOTICE.md](NOTICE.md)。
