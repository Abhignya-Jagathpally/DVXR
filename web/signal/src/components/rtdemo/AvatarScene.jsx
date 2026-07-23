// EXPERIMENTAL real-time avatar scene (react-three-fiber).
//
// A cube "avatar" translates per the decoded BCI command (Left/Right/Push/Pull), tints
// from calm (blue) to stressed (red) by the stress index, and shows a glucose halo. When
// the glucose channel abstains (the default, honest state) the halo is greyed and the
// overlay says "insufficient data" — never a fabricated value.

import { Canvas, useFrame } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";

const COMMAND_OFFSET = {
  Neutral: [0, 0, 0],
  Left: [-1.4, 0, 0],
  Right: [1.4, 0, 0],
  Push: [0, 0, -1.2],
  Pull: [0, 0, 1.2],
};

const CALM = new THREE.Color("#2563eb");
const STRESSED = new THREE.Color("#dc2626");

function Avatar({ frameRef }) {
  const mesh = useRef();
  const target = useRef(new THREE.Vector3());
  const color = useRef(new THREE.Color("#2563eb"));

  useFrame(() => {
    const frame = frameRef.current;
    if (!mesh.current || !frame) return;
    const offset = COMMAND_OFFSET[frame.command] || COMMAND_OFFSET.Neutral;
    target.current.set(offset[0], offset[1], offset[2]);
    // smooth translation toward the commanded pose
    mesh.current.position.lerp(target.current, 0.12);
    mesh.current.rotation.y += 0.01 + 0.03 * (frame.stress ?? 0);
    // tint by stress
    color.current.copy(CALM).lerp(STRESSED, Math.max(0, Math.min(1, frame.stress ?? 0)));
    mesh.current.material.color.copy(color.current);
  });

  return (
    <mesh ref={mesh} castShadow>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial metalness={0.2} roughness={0.4} />
    </mesh>
  );
}

function GlucoseHalo({ frameRef }) {
  const ring = useRef();
  useFrame(() => {
    const frame = frameRef.current;
    if (!ring.current || !frame) return;
    const abstained = frame.abstained !== false;
    ring.current.material.opacity = abstained ? 0.18 : 0.6;
    ring.current.material.color.set(abstained ? "#94a3b8" : "#12b76a");
    ring.current.rotation.z += 0.004;
  });
  return (
    <mesh ref={ring} rotation={[Math.PI / 2, 0, 0]} position={[0, -0.9, 0]}>
      <torusGeometry args={[1.8, 0.05, 12, 64]} />
      <meshBasicMaterial transparent opacity={0.2} color="#94a3b8" />
    </mesh>
  );
}

export default function AvatarScene({ frameRef }) {
  return (
    <Canvas
      camera={{ position: [3.2, 2.4, 4.2], fov: 50 }}
      style={{ width: "100%", height: "100%", display: "block" }}
      dpr={[1, 2]}
    >
      <color attach="background" args={["#0b1020"]} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 6, 4]} intensity={1.1} />
      <Avatar frameRef={frameRef} />
      <GlucoseHalo frameRef={frameRef} />
      <gridHelper args={[12, 12, "#1e293b", "#141c2f"]} position={[0, -1.4, 0]} />
    </Canvas>
  );
}
