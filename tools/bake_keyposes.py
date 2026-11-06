"""SOMA Fighter's clips.npz -> assets/keyposes.json: the few key frames this game generates toward.

clips.npz (from SOMA Fighter's tools/bake_clips.py) holds SOMA77 local poses in MotionBricks' runtime space
(Z up, each clip placed with its first frame at the origin facing +X). This keeps only the Mixamo-derived clips
(`fbx-*` sources), converts the chosen frames to SOMA model space (Y up) in this game's key-pose frame (facing
+Z, hips over the origin) and stores, per key, four consecutive poses as the hips position plus the world
rotations of the 30 model joints. The official SOMA BVH samples (NVIDIA Sample Data Evaluation License) and the
stills estimated from pictures are not used.

    uv run --with numpy --with scipy python -I tools/bake_keyposes.py <clips.npz> <clips.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

MODEL_JOINTS = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]

# key name -> (clip, first frame); four frames from there are kept (the key pose and the three after it)
KEYS = {
    "stance": ("stance", 0),
    "stance_b": ("stance", 50),
    "jab": ("jab", 5),
    "cross": ("jab_cross", 20),
    "hook": ("hook", 31),
    "uppercut": ("uppercut", 14),
    "mid_kick": ("mid_kick", 18),
    "high_kick": ("roundhouse", 32),
    "hit_head_light": ("hit_head_light", 8),
    "hit_head": ("hit_head_medium", 10),
    "hit_head_big": ("hit_head_big", 12),
    "hit_body": ("hit_body", 15),
    "hit_stomach": ("hit_stomach", 9),
    "down_back": ("down_back", 57),
    "down_ko": ("down_ko", 100),
    "victory": ("victory", 60),
    "victory_b": ("victory", 100),
    "taunt": ("taunt", 20),
    "taunt_b": ("taunt", 38),
    "block_high": ("block_high", 14),
    "block_mid": ("block_mid", 16),
}

# runtime (Z up, facing +X) -> model (Y up): undo MotionBricks' MUJOCO conversion (+90 deg about X),
# then turn +X to +Z so the key faces the opponent along +Z.
TO_MODEL = R.from_euler("y", -90, degrees=True) * R.from_euler("x", -90, degrees=True)


def fk(pose, parents):
    """(J, 7) local xyz + xyzw -> world rotations (J) and positions (J, 3)."""
    n = len(parents)
    rot = [None] * n
    pos = np.zeros((n, 3))
    for j in range(n):
        q = R.from_quat(pose[j, 3:])
        p = parents[j]
        if p < 0:
            rot[j], pos[j] = TO_MODEL * q, TO_MODEL.apply(pose[j, :3])
        else:
            rot[j], pos[j] = rot[p] * q, pos[p] + rot[p].apply(pose[j, :3])
    return rot, pos


def main(npz_path, json_path):
    clips = json.loads(Path(json_path).read_text(encoding="utf-8"))["clips"]
    data = np.load(npz_path, allow_pickle=False)
    names = [str(n) for n in data["names"]]
    parents = [int(p) for p in data["parents"]]
    idx = [names.index(n) for n in MODEL_JOINTS]
    out = {}
    for key, (clip, start) in KEYS.items():
        src = clips[clip]["source"]
        if not src.startswith("fbx-") or src.startswith("fbx-soma-poses"):
            raise SystemExit(f"{key}: {clip} comes from {src}, which this game does not ship")
        poses = data["clip_" + clip]
        frames = []
        origin = None
        for f in range(start, start + 4):
            rot, pos = fk(poses[min(f, len(poses) - 1)].astype(np.float64), parents)
            if origin is None:
                origin = np.array([pos[0, 0], 0.0, pos[0, 2]])
            hips = pos[0] - origin
            floor = pos[:, 1].min()
            frames.append({
                "root": [round(float(v), 4) for v in hips],
                "rot": [[round(float(v), 5) for v in rot[j].as_quat()] for j in idx],
                "floor": round(float(floor), 4),
            })
        out[key] = {"clip": clip, "frame": start, "source": src, "frames": frames}
        print(f"{key:15s} {clip:16s} {start:4d}  hips y {frames[0]['root'][1]:.2f}  lowest joint {frames[0]['floor']:+.3f}")
    doc = {
        "about": "Key poses for SOMA Fighter Web, converted from Mixamo-derived SOMA clips (SOMA Fighter clips.npz). "
                 "Space: SOMA model space, metres, Y up; each key faces +Z with the first pose's hips over the origin.",
        "joints": MODEL_JOINTS,
        "keys": out,
    }
    dest = Path(__file__).resolve().parents[1] / "assets" / "keyposes.json"
    dest.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
