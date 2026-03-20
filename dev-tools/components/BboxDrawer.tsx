"use client";

import { useRef, useState, useCallback, useEffect } from "react";

export interface Bbox {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Props {
  imageSrc: string;
  onBboxChange?: (bbox: Bbox | null) => void;
  existingBbox?: Bbox | null;
  /** Second bbox (for calibration: overlay=green, camera=blue) */
  secondBbox?: Bbox | null;
  bboxColor?: string;
  secondBboxColor?: string;
  className?: string;
}

export function BboxDrawer({
  imageSrc,
  onBboxChange,
  existingBbox,
  secondBbox,
  bboxColor = "#22c55e",
  secondBboxColor = "#3b82f6",
  className,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(null);
  const [currentBbox, setCurrentBbox] = useState<Bbox | null>(existingBbox || null);
  const [scale, setScale] = useState(1);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      const canvas = canvasRef.current;
      if (!canvas) return;
      // Fit to container width (max 800px)
      const maxW = Math.min(800, canvas.parentElement?.clientWidth || 800);
      const s = maxW / img.width;
      setScale(s);
      canvas.width = img.width * s;
      canvas.height = img.height * s;
      redraw(img, s, existingBbox || null, secondBbox || null);
    };
    img.src = imageSrc;
  }, [imageSrc, existingBbox, secondBbox]);

  const redraw = useCallback(
    (img: HTMLImageElement, s: number, bbox: Bbox | null, bbox2: Bbox | null) => {
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.drawImage(img, 0, 0, img.width * s, img.height * s);

      // Draw second bbox first (behind)
      if (bbox2) {
        ctx.strokeStyle = secondBboxColor;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(bbox2.x * s, bbox2.y * s, bbox2.w * s, bbox2.h * s);
        ctx.setLineDash([]);
      }
      // Draw primary bbox
      if (bbox) {
        ctx.strokeStyle = bboxColor;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(bbox.x * s, bbox.y * s, bbox.w * s, bbox.h * s);
        ctx.setLineDash([]);
      }
    },
    [bboxColor, secondBboxColor]
  );

  const getImgCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return {
      x: Math.round((e.clientX - rect.left) / scale),
      y: Math.round((e.clientY - rect.top) / scale),
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setDrawing(true);
    setStartPos(getImgCoords(e));
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing || !startPos || !imgRef.current) return;
    const pos = getImgCoords(e);
    const bbox: Bbox = {
      x: Math.min(startPos.x, pos.x),
      y: Math.min(startPos.y, pos.y),
      w: Math.abs(pos.x - startPos.x),
      h: Math.abs(pos.y - startPos.y),
    };
    setCurrentBbox(bbox);
    redraw(imgRef.current, scale, bbox, secondBbox || null);
  };

  const handleMouseUp = () => {
    setDrawing(false);
    setStartPos(null);
    onBboxChange?.(currentBbox);
  };

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ cursor: "crosshair" }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    />
  );
}
