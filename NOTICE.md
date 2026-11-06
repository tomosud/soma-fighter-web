# Notices

このリポジトリのコードは Apache License 2.0（[LICENSE](LICENSE)）です。次のものを含む・使います。

## NVIDIA MotionBricks（[NVlabs/motionbricks](https://github.com/NVlabs/motionbricks)、revision `54fc563`）

- **コード**（Apache-2.0）：`src/soma.js` の入力のエンコード・出力のデコード・端点補正は、`motionbricks/model/soma/kernels.py` と `motionbricks/model/kernels.py` を JavaScript に移したものです。
- **骨格**（CC-BY-4.0）：`assets/soma_rig.json` は `motionbricks/assets/human/soma_base_skel_minimal_no_root.usd` から `tools/extract_rig.py` で取り出した 30 関節のレスト姿勢です。© 2026 NVIDIA CORPORATION & AFFILIATES. 元のライセンス表記は [licenses/MotionBricks-LICENSE.txt](licenses/MotionBricks-LICENSE.txt)。
- **モデルの重み**（[NVIDIA Open Model Agreement](https://github.com/NVlabs/motionbricks/blob/54fc563aeaaf819a753d72cc9fdc6713989cdd43/motionbricks/assets/onnx/LICENSE.md)）：このリポジトリには含めません。プレイヤーのブラウザが NVIDIA の GitHub から `mmm_v6_soma_fp16.onnx` / `mmm_v6_soma_fp32.onnx` を直接ダウンロードします。Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement.
- 公式のサンプルモーション（NVIDIA Sample Data Evaluation License）は使っていません。

## キーポーズ（`assets/keyposes.json`）

SOMA Fighter の `data/clips.npz` のうち、Mixamo（Adobe）のアニメーションを Motion Bricks Studio で SOMA 骨格に移したクリップ（`fbx-*`）から、ゲームが目標にする 16 個のキー × 4 フレームだけを取り出したもの（`tools/bake_keyposes.py`）。元のクリップとフレームは JSON の各キーに記録してある。公式 SOMA サンプル（BVH）由来のクリップと、画像から推定した静止ポーズは含めていない。

## ライブラリ（CDN から読み込み）

- [three.js](https://github.com/mrdoob/three.js) 0.186.1 — MIT
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) 1.30.0 — MIT
