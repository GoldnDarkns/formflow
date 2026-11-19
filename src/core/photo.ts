// Finds the portrait photo on an identity document so a redacted copy can
// hide it. Classical computer vision, no model: the photo is the largest
// portrait-shaped block that is not text and does not look like the card's
// background. It is a guess that the person can switch off or redraw.
import type { Box, GrayImage } from './types';

export interface PhotoOptions {
  minArea: number; // share of the page
  maxArea: number;
  minAspect: number; // width / height
  maxAspect: number;
  minFill: number; // share of the bounding box that is "not background"
}
export const DEFAULT_PHOTO: PhotoOptions = { minArea: 0.02, maxArea: 0.35, minAspect: 0.5, maxAspect: 1.15, minFill: 0.7 };

const GRID = 160; // working width in cells

/** Box (in image pixels) of the most photo-like region, or null. `text` are the word boxes to ignore. */
export function findPhotoArea(g: GrayImage, text: Box[], opt: PhotoOptions = DEFAULT_PHOTO): Box | null {
  const s = Math.max(1, g.width / GRID);
  const w = Math.max(1, Math.round(g.width / s)), h = Math.max(1, Math.round(g.height / s));
  if (w < 16 || h < 16) return null;
  // box-average downsample: smooths away print texture and fine security patterns
  const sum = new Float64Array(w * h), cnt = new Uint32Array(w * h);
  for (let y = 0; y < g.height; y++) {
    const yy = Math.min(h - 1, Math.floor(y / s)) * w;
    for (let x = 0; x < g.width; x++) {
      const i = yy + Math.min(w - 1, Math.floor(x / s));
      sum[i] += g.data[y * g.width + x];
      cnt[i]++;
    }
  }
  const v = new Float64Array(w * h);
  for (let i = 0; i < v.length; i++) v[i] = cnt[i] ? sum[i] / cnt[i] : 255;

  // cells covered by text are never part of the photo
  const isText = new Uint8Array(w * h);
  for (const b of text) {
    const x0 = Math.max(0, Math.floor(b[0] / s) - 1), x1 = Math.min(w - 1, Math.ceil(b[2] / s));
    const y0 = Math.max(0, Math.floor(b[1] / s) - 1), y1 = Math.min(h - 1, Math.ceil(b[3] / s));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) isText[y * w + x] = 1;
  }

  // background level = median of the non-text cells; spread = median absolute deviation
  const free: number[] = [];
  for (let i = 0; i < v.length; i++) if (!isText[i]) free.push(v[i]);
  if (free.length < 50) return null;
  free.sort((a, b) => a - b);
  const bg = free[free.length >> 1];
  const dev = free.map((x) => Math.abs(x - bg)).sort((a, b) => a - b);
  const mad = dev[dev.length >> 1];
  const thr = Math.max(8, 4 * 1.4826 * mad);

  const on = new Uint8Array(w * h);
  for (let i = 0; i < v.length; i++) on[i] = !isText[i] && Math.abs(v[i] - bg) > thr ? 1 : 0;

  // connected components (4-neighbours); text cells inside a photo block are bridged
  const seen = new Uint8Array(w * h);
  let best: { box: Box; score: number } | null = null;
  const stack: number[] = [];
  for (let start = 0; start < on.length; start++) {
    if (!on[start] || seen[start]) continue;
    let x0 = w, y0 = h, x1 = 0, y1 = 0;
    stack.push(start);
    seen[start] = 1;
    while (stack.length) {
      const i = stack.pop()!;
      const x = i % w, y = (i - x) / w;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
        if (j >= 0 && on[j] && !seen[j]) { seen[j] = 1; stack.push(j); }
      }
    }
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    const area = (bw * bh) / (w * h), aspect = bw / bh;
    // text printed over the block (a "PHOTO" watermark, a name) counts as filled
    let filled = 0;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (on[y * w + x] || isText[y * w + x]) filled++;
    const fill = filled / (bw * bh);
    if (area < opt.minArea || area > opt.maxArea || aspect < opt.minAspect || aspect > opt.maxAspect || fill < opt.minFill) continue;
    // a block touching every edge is the page itself, not a photo
    if (x0 === 0 && y0 === 0 && x1 === w - 1 && y1 === h - 1) continue;
    const score = area * fill;
    if (!best || score > best.score) best = { box: [x0 * s, y0 * s, Math.min(g.width, (x1 + 1) * s), Math.min(g.height, (y1 + 1) * s)], score };
  }
  return best ? (best.box.map(Math.round) as Box) : null;
}
