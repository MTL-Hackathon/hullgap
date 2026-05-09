"use client";

import { useRef, useEffect, useCallback } from "react";

const NODE_COUNT = 100;
const NODE_SIZE = 6;
const CONNECTION_DIST = 160;
const MOUSE_RADIUS = 140;
const MOUSE_STRENGTH = 0.8;
const DRIFT_SPEED = 0.3;
const DAMPING = 0.95;
const ACCENT = { r: 40, g: 159, b: 240 }; // #289ff0

interface Node {
  x: number;
  y: number;
  baseX: number;
  baseY: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  driftAngle: number;
  driftSpeed: number;
}

interface Props {
  mouseX?: number;
  mouseY?: number;
  isHovered?: boolean;
}

export function CrystalLatticeBg({
  mouseX = 0,
  mouseY = 0,
  isHovered = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<Node[]>([]);
  const rafRef = useRef<number>(0);
  const sizeRef = useRef({ w: 0, h: 0 });
  const mouseRef = useRef({ x: 0, y: 0, active: false });

  const initNodes = useCallback((w: number, h: number) => {
    const cols = Math.ceil(Math.sqrt(NODE_COUNT * (w / h)));
    const rows = Math.ceil(NODE_COUNT / cols);
    const spacingX = w / cols;
    const spacingY = h / rows;
    const nodes: Node[] = [];

    for (let i = 0; i < NODE_COUNT; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const jitterX = (Math.random() - 0.5) * spacingX * 0.6;
      const jitterY = (Math.random() - 0.5) * spacingY * 0.6;
      const x = spacingX * (col + 0.5) + jitterX;
      const y = spacingY * (row + 0.5) + jitterY;

      nodes.push({
        x,
        y,
        baseX: x,
        baseY: y,
        vx: 0,
        vy: 0,
        size: NODE_SIZE + (Math.random() - 0.5) * 2,
        opacity: 0.55 + Math.random() * 0.35,
        driftAngle: Math.random() * Math.PI * 2,
        driftSpeed: DRIFT_SPEED * (0.5 + Math.random() * 0.5),
      });
    }

    nodesRef.current = nodes;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      sizeRef.current = { w: rect.width, h: rect.height };

      if (nodesRef.current.length === 0) {
        initNodes(rect.width, rect.height);
      }
    };

    resize();
    window.addEventListener("resize", resize);

    let time = 0;

    const draw = () => {
      const { w, h } = sizeRef.current;
      const nodes = nodesRef.current;
      const mouse = mouseRef.current;

      ctx.clearRect(0, 0, w, h);
      time += 0.016;

      for (const node of nodes) {
        node.driftAngle += 0.003;
        const driftX = Math.cos(node.driftAngle) * node.driftSpeed;
        const driftY = Math.sin(node.driftAngle * 0.7 + 1.2) * node.driftSpeed;
        node.baseX += driftX * 0.15;
        node.baseY += driftY * 0.15;

        if (node.baseX < -20) node.baseX = w + 20;
        if (node.baseX > w + 20) node.baseX = -20;
        if (node.baseY < -20) node.baseY = h + 20;
        if (node.baseY > h + 20) node.baseY = -20;

        let fx = 0;
        let fy = 0;

        if (mouse.active) {
          const dx = node.x - mouse.x;
          const dy = node.y - mouse.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < MOUSE_RADIUS && dist > 0.1) {
            const falloff = 1 - dist / MOUSE_RADIUS;
            const force = MOUSE_STRENGTH * falloff * falloff;
            fx += (dx / dist) * force;
            fy += (dy / dist) * force;
          }
        }

        const restoreX = (node.baseX - node.x) * 0.02;
        const restoreY = (node.baseY - node.y) * 0.02;

        node.vx = (node.vx + fx + restoreX) * DAMPING;
        node.vy = (node.vy + fy + restoreY) * DAMPING;
        node.x += node.vx;
        node.y += node.vy;
      }

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECTION_DIST) {
            const alpha = (1 - dist / CONNECTION_DIST) * 0.45;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = `rgba(${ACCENT.r}, ${ACCENT.g}, ${ACCENT.b}, ${alpha})`;
            ctx.lineWidth = 1.2;
            ctx.stroke();
          }
        }
      }

      for (const node of nodes) {
        const pulse = Math.sin(time * 1.2 + node.driftAngle) * 0.1;
        const alpha = node.opacity + pulse;
        const half = node.size / 2;

        ctx.fillStyle = `rgba(${ACCENT.r}, ${ACCENT.g}, ${ACCENT.b}, ${alpha})`;
        ctx.fillRect(node.x - half, node.y - half, node.size, node.size);
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [initNodes]);

  useEffect(() => {
    if (!isHovered) {
      mouseRef.current.active = false;
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    mouseRef.current = {
      x: ((mouseX + 1) / 2) * rect.width,
      y: ((1 - mouseY) / 2) * rect.height,
      active: true,
    };
  }, [mouseX, mouseY, isHovered]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}
