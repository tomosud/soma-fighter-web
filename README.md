# SOMA Fighter Web

NVIDIA [MotionBricks](https://github.com/NVlabs/motionbricks)（SOMA モデル）を**ブラウザの WebGPU で動かし**、その場で生成した動きだけで戦う 1 対 1 の格闘ゲーム。サーバーは要らず、GitHub Pages の静的ファイルだけで動く。

用意したアニメーションは再生しない。構え・打撃 6 種・のけぞり 5 種・ダウン 2 種・勝ちポーズの**キーポーズだけ**（Mixamo のモーションを SOMA に移したものから、当たる瞬間などの 4 フレームずつ。`assets/keyposes.json`）を相手の今の位置に合わせて置き（打撃は IK で手足の先を相手の頭・あご・胴へ届かせ）、そこまでの動きを MotionBricks が毎回生成する。2 人分の生成は 1 回の推論にまとめる。

デスクトップ版 SOMA Fighter（TensorRT サーバー + ブラウザ描画）の考え方を参考にした、最小限のブラウザ版。

## 遊ぶ

GitHub Pages の URL を開き、「モデルを読み込んで始める」を押す。初回は NVIDIA の GitHub からモデル（Mixed FP16 で 380 MB）をダウンロードしてブラウザ（IndexedDB）に保存する（2 回目からは保存したものを使う）。

| 操作 | キー |
|---|---|
| 前後に動く | A / D（矢印も可） |
| 横に動く（奥 / 手前） | W / S |
| ジャブ | J（左クリック） |
| ストレート | L（中クリック） |
| フック | U |
| アッパー | I |
| ミドルキック | K（右クリック） |
| ハイキック | O |

2 本先取。届かない距離で打つと空振りになる。打たれると打撃は止まる。

### 必要なもの

- WebGPU が使えるデスクトップの Chrome / Edge（WebGPU が無いと WASM（CPU）で動くが、かなり遅い）
- 数 GB の空きメモリ

### 測った値

| 環境 | 1 回の生成（2 人分） |
|---|---|
| WebGPU、NVIDIA Turing（Claude デスクトップの Chromium 152） | 約 120〜170 ms |
| Node の onnxruntime（CPU） | 約 80 ms |

バッチの大きさを 2 に固定して ONNX Runtime に形の計算を畳ませると、可変のときより約 3 割速い（WebGPU で約 180〜230 ms → 120〜170 ms）。生成にかかる時間のぶん先の姿勢から生成を始め、届いたら続きとしてつなぐ（`src/motion.js`）。間に合わなければ 0.18 秒で混ぜてつなぐ。

## SOMA Pull（`lab.html`）

人形の手・腰・頭・足をつかんで引っ張ると、人間らしい足運びでついてくる。

1. **ひも**：つかんだ点は、ひも（手なら腕の長さ）が張ってから体を引く。張った長さで行きたい速さが決まり（上限 1〜3 m/s、加速も制限）、腰は上へ上がらないので跳ばない。体は進む向きへ回り（最大 3 rad/s）、軽く後ろへ引かれたときは後ずさりする。
2. **約 1 秒先の、動いている途中の目標**：予測した経路の約 1 秒先に、1/30 秒おきの目標を 4 枚置く。目標が速さと向きを持つので、MotionBricks は踏み込んで止まるのではなく、自分で足を踏み替えて歩く・走る。0.3 秒ごとに、結果が届く時点の姿勢から作り直し、等速で再生する。
3. **モデル自身の出力から育つ歩きのポーズ集**：生成された動きのうち、速さが一定で足が床近くにある 4 フレームを、速さ（ゆっくり・歩き・早歩き・走り）× 向き × どちらの足が着いているか（モデルの接地の出力）で溜め、次の目標に使う。最初は立ち姿勢を経路に置くだけだが、使うほど本物の歩き・走りの姿勢がたまる。
4. **画面の上の仕上げ**：つかんだ手・足は IK で指の位置へ寄せ（届く範囲で）、モデルが「着いている」とした足は着いた場所に止めて滑らせない。

測った値（右手を 1.2 m/s で 7 秒引く／腰を 2.5 m/s で 5 秒引く）：体の速さ 0.99 / 1.8 m/s、歩数 1.86 / 2.6 歩/秒、両足が 18cm より上がる時間 0% / 0%、腰の高さ 0.87〜1.03 / 0.86〜1.00 m、着いている足の滑り 0.4 / 0.2 mm/フレーム。

（前の版の、姿勢を自動で変異させて展開する「Pose Lab」はコミット `e0a6d3f` までの履歴にある。）

## 仕組み

```
src/
  soma.js          骨格（30 関節）、順運動学、MotionBricks の入力エンコード・出力デコード・端点補正
  poses.js         キーポーズ：keyposes.json の読み込み、打撃を点へ届かせる 2 関節 IK（KeyBook）。手で作る版（PoseBook）も残す
  motion.js        Track（再生中の動きと、生成した動きのつなぎ方）と Scheduler（2 人分を 1 回にまとめる）
  infer.worker.js  ONNX Runtime Web（WebGPU、だめなら WASM）、モデルのダウンロードと IndexedDB への保存（Blob 1 つ）
  render.js        three.js：カプセルの人形、床、2 人を横から映すカメラ
  main.js          試合：入力、CPU、当たり判定、ラウンド
  lab.js           SOMA Pull：ひもの移動制御、約 1 秒先の目標、育つ歩きのポーズ集、手の IK と足の固定
assets/soma_rig.json   公式の骨格 USD から取り出したレスト姿勢
assets/keyposes.json   キーポーズ（SOMA Fighter の clips.npz のうち Mixamo 由来のクリップから）
tools/
  extract_rig.py   USD → soma_rig.json（uv run --with usd-core python tools/extract_rig.py <usd>）
  bake_keyposes.py clips.npz → keyposes.json（uv run --with numpy --with scipy python -I tools/bake_keyposes.py <clips.npz> <clips.json>）
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

コードは Apache-2.0。MotionBricks のコード（Apache-2.0）を移した部分、骨格データ（CC-BY-4.0）、キーポーズ（Mixamo 由来）、モデルの重み（NVIDIA Open Model Agreement、このリポジトリには含めない）については [NOTICE.md](NOTICE.md)。
