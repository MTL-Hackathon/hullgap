"use client";

import { useSyncExternalStore } from "react";
import { Canvas } from "@react-three/fiber";
import { CrystalParticles } from "./crystal-particles";

const subscribeNoop = () => () => {};
const getClientSnapshot = () => true;
const getServerSnapshot = () => false;

interface Props {
  mouseX?: number;
  mouseY?: number;
  isHovered?: boolean;
}

export function HeroSphere({ mouseX = 0, mouseY = 0, isHovered = false }: Props) {
  const mounted = useSyncExternalStore(
    subscribeNoop,
    getClientSnapshot,
    getServerSnapshot
  );
  if (!mounted) return null;

  return (
    <Canvas
      frameloop="always"
      camera={{ position: [0, 0, 5], fov: 75 }}
      gl={{ antialias: true, alpha: true }}
      style={{ background: "transparent", width: "100%", height: "100%" }}
    >
      <ambientLight intensity={0.4} />
      <CrystalParticles mouseX={mouseX} mouseY={mouseY} isHovered={isHovered} />
    </Canvas>
  );
}
