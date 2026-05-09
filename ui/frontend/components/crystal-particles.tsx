"use client";

import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

const COUNT = 2500;
const RADIUS = 1.8;
const AMP = 2.5;
const EXPAND_TIME = 5;
const HOLD_TIME = 2;
const DEFLATE_TIME = 5;
const TOTAL_PERIOD = EXPAND_TIME + HOLD_TIME + DEFLATE_TIME;

function jitter(i: number, salt: number): number {
  const x = Math.sin(i * 12.9898 + salt * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

interface Props {
  mouseX?: number;
  mouseY?: number;
  isHovered?: boolean;
}

export function CrystalParticles({ mouseX = 0, mouseY = 0, isHovered = false }: Props) {
  const ref = useRef<THREE.Points>(null);
  const t = useRef(0);
  const pVel = useRef(new Float32Array(COUNT * 3).fill(0));
  const pPos = useRef(new Float32Array(COUNT * 3).fill(0));

  const { sphere, geomPos, colBuf } = useMemo(() => {
    const sphere = new Float32Array(COUNT * 3);
    const geomPos = new Float32Array(COUNT * 3);
    const colBuf = new Float32Array(COUNT * 3);

    for (let i = 0; i < COUNT; i++) {
      const i3 = i * 3;
      const phi = Math.acos(1 - (2 * (i + 0.5)) / COUNT);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;

      const x = RADIUS * Math.sin(phi) * Math.cos(theta);
      const y = RADIUS * Math.sin(phi) * Math.sin(theta);
      const z = RADIUS * Math.cos(phi);

      sphere[i3] = geomPos[i3] = x + (jitter(i, 1) - 0.5) * 0.07;
      sphere[i3 + 1] = geomPos[i3 + 1] = y + (jitter(i, 2) - 0.5) * 0.07;
      sphere[i3 + 2] = geomPos[i3 + 2] = z + (jitter(i, 3) - 0.5) * 0.07;

      // Blue gradient: #289ff0 → lighter cyan
      const mix = i / COUNT;
      colBuf[i3] = 0.16 + mix * 0.3;       // R
      colBuf[i3 + 1] = 0.62 + mix * 0.2;   // G
      colBuf[i3 + 2] = 0.94;               // B
    }

    return { sphere, geomPos, colBuf };
  }, []);

  useFrame((_, delta) => {
    if (!ref.current) return;

    t.current += delta;

    const pos = ref.current.geometry.attributes.position.array as Float32Array;
    const col = ref.current.geometry.attributes.color.array as Float32Array;
    const pv = pVel.current;
    const pp = pPos.current;

    const phase = t.current % TOTAL_PERIOD;
    let wave: number;
    if (phase < EXPAND_TIME) {
      wave = (1 - Math.cos((Math.PI * phase) / EXPAND_TIME)) / 2;
    } else if (phase < EXPAND_TIME + HOLD_TIME) {
      wave = 1;
    } else {
      const p = phase - EXPAND_TIME - HOLD_TIME;
      wave = (1 + Math.cos((Math.PI * p) / DEFLATE_TIME)) / 2;
    }
    const scale = 0.5 + AMP * wave;

    const mx = mouseX * 3.84;
    const my = mouseY * 3.84;

    for (let i = 0; i < COUNT; i++) {
      const i3 = i * 3;

      const bx = sphere[i3] * scale;
      const by = sphere[i3 + 1] * scale;
      const bz = sphere[i3 + 2] * scale;

      let fx = 0;
      let fy = 0;
      let fz = 0;
      if (isHovered) {
        const dx = bx - mx;
        const dy = by - my;
        const dz = bz;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const infR = 2.2 * scale;
        if (dist < infR && dist > 0.001) {
          const fall = 1 - dist / infR;
          const strength = 0.045 * fall * fall;
          fx = (dx / dist) * strength;
          fy = (dy / dist) * strength;
          fz = (dz / dist) * strength * 0.25;
        }
      }

      const springBack = 0.03;
      pv[i3] = (pv[i3] + fx - pp[i3] * springBack) * 0.91;
      pv[i3 + 1] = (pv[i3 + 1] + fy - pp[i3 + 1] * springBack) * 0.91;
      pv[i3 + 2] = (pv[i3 + 2] + fz - pp[i3 + 2] * springBack) * 0.91;

      pp[i3] += pv[i3];
      pp[i3 + 1] += pv[i3 + 1];
      pp[i3 + 2] += pv[i3 + 2];

      pos[i3] = bx + pp[i3];
      pos[i3 + 1] = by + pp[i3 + 1];
      pos[i3 + 2] = bz + pp[i3 + 2];

      // Animate color: blue shimmer
      const s = Math.sin(t.current * 0.28 + i * 0.09) * 0.5 + 0.5;
      col[i3] = 0.16 + s * 0.3;
      col[i3 + 1] = 0.62 + s * 0.2;
      col[i3 + 2] = 0.94;
    }

    ref.current.rotation.y += delta * 0.04;

    ref.current.geometry.attributes.position.needsUpdate = true;
    ref.current.geometry.attributes.color.needsUpdate = true;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[geomPos, 3]} />
        <bufferAttribute attach="attributes-color" args={[colBuf, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.03}
        vertexColors
        transparent
        opacity={0.9}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}
