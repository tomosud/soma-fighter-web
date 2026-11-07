// three.js scene: capsule mannequins, the floor and a side camera that keeps both fighters in view.
import * as THREE from 'three';

// [from, to, radius] by joint name ('@' marks an extra end joint)
const BONES = [
  ['Hips', 'Spine2', 0.13], ['Spine2', 'Chest', 0.14], ['Chest', 'Neck1', 0.13], ['Neck1', 'Head', 0.05],
  ['LeftLeg', 'RightLeg', 0.1],
  ['LeftShoulder', 'LeftArm', 0.055], ['LeftArm', 'LeftForeArm', 0.055], ['LeftForeArm', 'LeftHand', 0.045], ['LeftHand', 'LeftHandMiddleEnd', 0.048],
  ['RightShoulder', 'RightArm', 0.055], ['RightArm', 'RightForeArm', 0.055], ['RightForeArm', 'RightHand', 0.045], ['RightHand', 'RightHandMiddleEnd', 0.048],
  ['LeftLeg', 'LeftShin', 0.075], ['LeftShin', 'LeftFoot', 0.058], ['LeftFoot', 'LeftToeBase', 0.045], ['LeftToeBase', '@LeftToeEnd', 0.035],
  ['RightLeg', 'RightShin', 0.075], ['RightShin', 'RightFoot', 0.058], ['RightFoot', 'RightToeBase', 0.045], ['RightToeBase', '@RightToeEnd', 0.035],
];
const Y = new THREE.Vector3(0, 1, 0);

export class Mannequin {
  constructor(scene, rig, color) {
    this.rig = rig;
    this.material = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05 });
    this.flash = 0;
    const P = rig.positions(rig.restPose());
    const X = rig.extraPositions(rig.restPose(), P);
    const pick = (name, P, X) => (name[0] === '@' ? X[rig.extras.findIndex(e => e.name === name.slice(1))] : P[rig.index[name]]);
    this.pick = pick;
    this.group = new THREE.Group();
    this.bones = BONES.map(([a, b, r]) => {
      const len = pick(a, P, X).distanceTo(pick(b, P, X));
      // (each bone its own material: a striking limb can glow on its own)
      const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(r, Math.max(0.001, len), 4, 10), this.material.clone());
      mesh.castShadow = true;
      this.group.add(mesh);
      return { a, b, mesh };
    });
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.115, 20, 14), this.material.clone());
    this.tint = 0; this.limb = null; this.limbGlow = 0; // (a wind-up: the body glows orange, the striking limb more)
    this.head.castShadow = true;
    this.group.add(this.head);
    scene.add(this.group);
  }

  update(P, X, dt) {
    const tmp = new THREE.Vector3();
    for (const { a, b, mesh } of this.bones) {
      const pa = this.pick(a, P, X), pb = this.pick(b, P, X);
      mesh.position.addVectors(pa, pb).multiplyScalar(0.5);
      tmp.subVectors(pb, pa);
      if (tmp.lengthSq() > 1e-10) mesh.quaternion.setFromUnitVectors(Y, tmp.normalize());
    }
    const head = P[this.rig.index.Head], end = X[0];
    this.head.position.addVectors(head, end).multiplyScalar(0.5);
    this.flash = Math.max(0, this.flash - dt * 4);
    const er = this.flash * 0.9 + this.tint * 0.5, eg = this.flash * 0.25 + this.tint * 0.16, eb = this.flash * 0.1;
    const chain = this.limb ? (/Hand|ForeArm/.test(this.limb) ? [this.limb.replace(/Hand|ForeArm/, 'ForeArm'), this.limb.replace(/Hand|ForeArm/, 'Hand')] : [this.limb.replace(/Foot|Shin/, 'Shin'), this.limb.replace(/Foot|Shin/, 'Foot')]) : [];
    for (const { a, b, mesh } of this.bones) {
      const on = chain.includes(a) || chain.includes(b);
      mesh.material.emissive.setRGB(er + (on ? this.limbGlow : 0), eg + (on ? this.limbGlow * 0.4 : 0), eb);
    }
    this.head.material.emissive.setRGB(er, eg, eb);
  }
}

export class Stage {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14161c);
    this.scene.fog = new THREE.Fog(0x14161c, 9, 22);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 60);
    this.scene.add(new THREE.HemisphereLight(0xcfd8ff, 0x302820, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(2, 7, 3);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 0.5, far: 20 });
    this.scene.add(sun, sun.target);
    this.sun = sun;

    const floor = new THREE.Mesh(new THREE.CircleGeometry(12, 64), new THREE.MeshStandardMaterial({ color: 0x2a2d36, roughness: 0.95 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
    this.floor = floor;
    const ring = new THREE.Mesh(new THREE.RingGeometry(3.95, 4.05, 96), new THREE.MeshBasicMaterial({ color: 0xc8a24a }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.002;
    this.scene.add(ring);
    this.ring = ring;
    const grid = new THREE.GridHelper(8, 16, 0x3b404d, 0x31353f);
    grid.position.y = 0.001;
    this.scene.add(grid);
    this.grid = grid;

    this.side = 1;
    this.camPos = new THREE.Vector3(0, 1.6, 5);
    this.camLook = new THREE.Vector3(0, 1, 0);
    this.resize();
    addEventListener('resize', () => this.resize());
  }

  resize() {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Side view across the line between the fighters; p1 stays on the left of the screen. */
  follow(a, b, dt) {
    const mid = new THREE.Vector3((a.x + b.x) / 2, 0, (a.z + b.z) / 2);
    const axis = new THREE.Vector3(b.x - a.x, 0, b.z - a.z);
    const sep = axis.length();
    if (sep > 1e-3) axis.divideScalar(sep); else axis.set(1, 0, 0);
    const normal = new THREE.Vector3(-axis.z, 0, axis.x); // camera side: p1 on the left, p2 on the right
    const dist = Math.max(3.4, sep * 1.25 + 2.4);
    const want = mid.clone().addScaledVector(normal, dist).setY(1.55);
    const k = 1 - Math.exp(-dt * 4);
    this.camPos.lerp(want, k);
    this.camLook.lerp(mid.clone().setY(1.0), k);
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.camLook);
    this.sun.position.set(mid.x + 2, 7, mid.z + 3);
    this.sun.target.position.copy(mid);
    return { right: axis, toward: normal };
  }

  /** An endless floor: a large floor and a grid that move with `center`, the grid in whole cells so it looks fixed. */
  makeEndless() {
    this.scene.remove(this.floor, this.ring, this.grid);
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ color: 0x2a2d36, roughness: 0.95 }));
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.receiveShadow = true;
    this.grid = new THREE.GridHelper(80, 80, 0x3b404d, 0x31353f);
    this.grid.position.y = 0.001;
    this.marks = new THREE.GridHelper(80, 8, 0x555b6b, 0x555b6b); // every 10 m, a little brighter
    this.marks.position.y = 0.002;
    this.scene.fog = new THREE.Fog(0x14161c, 14, 38);
    this.scene.add(this.floor, this.grid, this.marks);
  }

  followFloor(center) {
    this.floor.position.set(center.x, 0, center.z);
    this.grid.position.set(Math.round(center.x), 0.001, Math.round(center.z));
    this.marks.position.set(Math.round(center.x / 10) * 10, 0.002, Math.round(center.z / 10) * 10);
    this.sun.position.set(center.x + 2, 7, center.z + 3);
    this.sun.target.position.set(center.x, 0, center.z);
  }

  render() { this.renderer.render(this.scene, this.camera); }
}
