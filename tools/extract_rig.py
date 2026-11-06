"""Extract the 30 MotionBricks SOMA model joints from the official skeleton USD into assets/soma_rig.json.

Source: NVlabs/motionbricks `motionbricks/assets/human/soma_base_skel_minimal_no_root.usd` (Apache-2.0).
The output keeps, for each model joint, its nearest model ancestor and its rest pose in world space
(metres, Y up, rotations as xyzw quaternions). A few non-model end joints are kept for drawing.

    uv run --with usd-core python tools/extract_rig.py <path to soma_base_skel_minimal_no_root.usd>
"""
import json
import sys
from pathlib import Path

from pxr import Gf, Usd, UsdSkel

MODEL_JOINTS = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]
EXTRA_JOINTS = ["HeadEnd", "LeftToeEnd", "RightToeEnd"]


def main(usd_path: str) -> None:
    stage = Usd.Stage.Open(usd_path)
    scale = float(stage.GetMetadata("metersPerUnit") or 1.0)
    skel = next(UsdSkel.Skeleton(p) for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton))
    paths = [str(j) for j in skel.GetJointsAttr().Get()]
    rest = skel.GetRestTransformsAttr().Get()
    names = [p.split("/")[-1] for p in paths]
    parent = [paths.index(p.rsplit("/", 1)[0]) if "/" in p else -1 for p in paths]

    world = []
    for i, m in enumerate(rest):
        world.append(m if parent[i] < 0 else m * world[parent[i]])  # USD: row vectors, local * parentWorld

    def rest_of(i):
        m = Gf.Matrix4d(world[i])
        t = m.ExtractTranslation() * scale
        q = m.RemoveScaleShear().ExtractRotationQuat()
        im = q.GetImaginary()
        return [round(t[0], 6), round(t[1], 6), round(t[2], 6)], [round(im[0], 7), round(im[1], 7), round(im[2], 7), round(q.GetReal(), 7)]

    def nearest(i, allowed):
        p = parent[i]
        while p >= 0 and names[p] not in allowed:
            p = parent[p]
        return names[p] if p >= 0 else None

    joints = []
    for name in MODEL_JOINTS:
        i = names.index(name)
        pos, rot = rest_of(i)
        anc = nearest(i, MODEL_JOINTS)
        joints.append({"name": name, "parent": MODEL_JOINTS.index(anc) if anc else -1, "position": pos, "rotation": rot})
    extras = []
    for name in EXTRA_JOINTS:
        i = names.index(name)
        pos, rot = rest_of(i)
        extras.append({"name": name, "parent": MODEL_JOINTS.index(nearest(i, MODEL_JOINTS)), "position": pos})

    out = {
        "source": "NVlabs/motionbricks motionbricks/assets/human/soma_base_skel_minimal_no_root.usd (Apache-2.0)",
        "space": "SOMA model space: metres, Y up, rest pose (T-pose facing +Z)",
        "joints": joints,
        "extras": extras,
    }
    dest = Path(__file__).resolve().parents[1] / "assets" / "soma_rig.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {dest} ({len(joints)} joints, {len(extras)} extras)")


if __name__ == "__main__":
    main(sys.argv[1])
