// "Magic scan": what phone scanner apps do before any reading happens.
//   1. find the sheet of paper in a cluttered photo (four corners)
//   2. warp it flat with a perspective transform (homography)
//   3. even out the lighting so shadows and dark corners become white paper
// Pure typed-array code: runs in the Web Worker and in the Node test scripts.
import type { GrayImage } from './types';
import { otsu } from './preprocess';

export type Pt = [number, number];
export type Quad = [Pt, Pt, Pt, Pt]; // top-left, top-right, bottom-right, bottom-left

function boxBlur(g: GrayImage, r: number): GrayImage {
  const { width: w, height: h } = g;
  const tmp = new Float32Array(w * h), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = -r; x < w; x++) {
      if (x + r < w) s += g.data[y * w + x + r];
      if (x - r - 1 >= 0) s -= g.data[y * w + x - r - 1];
      if (x >= 0) tmp[y * w + x] = s / (Math.min(w - 1, x + r) - Math.max(0, x - r) + 1);
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y < h; y++) {
      if (y + r < h) s += tmp[(y + r) * w + x];
      if (y - r - 1 >= 0) s -= tmp[(y - r - 1) * w + x];
      if (y >= 0) out[y * w + x] = s / (Math.min(h - 1, y + r) - Math.max(0, y - r) + 1);
    }
  }
  return { width: w, height: h, data: out };
}

const quadArea = (q: Quad) => {
  let a = 0;
  for (let i = 0; i < 4; i++) a += q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1];
  return Math.abs(a) / 2;
};

/**
 * Shrink by averaging every block of pixels. Picking single pixels (as `downscale` does) keeps all
 * the grain of a textured surface, and the page outline then breaks up; the mean of a block does not.
 */
function shrinkMean(g: GrayImage, max: number): { img: GrayImage; scale: number } {
  const s = Math.min(1, max / Math.max(g.width, g.height));
  if (s === 1) return { img: g, scale: 1 };
  const w = Math.max(1, Math.round(g.width * s)), h = Math.max(1, Math.round(g.height * s));
  const sum = new Float64Array(w * h), cnt = new Uint32Array(w * h);
  for (let y = 0; y < g.height; y++) {
    const row = Math.min(h - 1, Math.floor(y * s)) * w;
    for (let x = 0; x < g.width; x++) { const i = row + Math.min(w - 1, Math.floor(x * s)); sum[i] += g.data[y * g.width + x]; cnt[i]++; }
  }
  const data = new Uint8Array(w * h);
  for (let i = 0; i < data.length; i++) data[i] = cnt[i] ? sum[i] / cnt[i] : 255;
  return { img: { width: w, height: h, data }, scale: s };
}

/** Convex hull of a set of points (monotone chain), counter-clockwise in image coordinates. */
function convexHull(pts: Pt[]): Pt[] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list: Pt[]) => { const hll: Pt[] = []; for (const q of list) { while (hll.length >= 2 && cross(hll[hll.length - 2], hll[hll.length - 1], q) <= 0) hll.pop(); hll.push(q); } hll.pop(); return hll; };
  return [...half(p), ...half(p.reverse())];
}

/**
 * The four corners of a page from the outline of its region: the hull's long edges give the
 * direction of each of the four sides, each side is the line of that direction touching the
 * hull, and the corners are where neighbouring sides cross. Null when four sides cannot be told apart.
 */
function sidesQuad(outline: Pt[], w: number, h: number): Quad | null {
  const hull = convexHull(outline);
  if (hull.length < 4) return null;
  const edges = hull.map((a, i) => { const b = hull[(i + 1) % hull.length]; const dx = b[0] - a[0], dy = b[1] - a[1]; return { a, b, len: Math.hypot(dx, dy), ang: Math.atan2(dy, dx) }; });
  const main = edges.reduce((m, e) => (e.len > m.len ? e : m)).ang;
  const cx = hull.reduce((t, p) => t + p[0], 0) / hull.length, cy = hull.reduce((t, p) => t + p[1], 0) / hull.length;
  // side k = edges running along main + k * 90 degrees (the hull is walked one way, so each side has its own direction)
  const lines: { nx: number; ny: number; c: number }[] = [];
  for (let k = 0; k < 4; k++) {
    const want = main + (k * Math.PI) / 2;
    let sx = 0, sy = 0, total = 0;
    for (const e of edges) {
      const d = Math.atan2(Math.sin(e.ang - want), Math.cos(e.ang - want));
      if (Math.abs(d) > (25 * Math.PI) / 180) continue;
      // long edges are real sides, short ones are the ragged bits; and a real side runs square to the
      // longest one, while the edge bridging a missing corner runs well off it
      const wgt = e.len * e.len * Math.exp(-((d / 0.12) ** 2));
      sx += Math.cos(want + d) * wgt; sy += Math.sin(want + d) * wgt; total += e.len;
    }
    if (total < Math.min(w, h) * 0.08) return null;
    const n = Math.hypot(sx, sy), dx = sx / n, dy = sy / n;
    // outward normal, and the line of this direction that touches the hull
    let nx = dy, ny = -dx;
    if (nx * (hull[0][0] - cx) + ny * (hull[0][1] - cy) < 0 && hull.every((p) => nx * (p[0] - cx) + ny * (p[1] - cy) <= 0)) { nx = -nx; ny = -ny; }
    const side = edges.filter((e) => Math.abs(Math.atan2(Math.sin(e.ang - want), Math.cos(e.ang - want))) <= (25 * Math.PI) / 180);
    const mid = side.reduce((t, e) => t + (nx * ((e.a[0] + e.b[0]) / 2 - cx) + ny * ((e.a[1] + e.b[1]) / 2 - cy)) * e.len, 0);
    if (mid < 0) { nx = -nx; ny = -ny; }
    lines.push({ nx, ny, c: Math.max(...hull.map((p) => nx * p[0] + ny * p[1])) });
  }
  const meet = (p: (typeof lines)[0], q: (typeof lines)[0]): Pt | null => { const det = p.nx * q.ny - p.ny * q.nx; return Math.abs(det) < 1e-6 ? null : [(p.c * q.ny - p.ny * q.c) / det, (p.nx * q.c - p.c * q.nx) / det]; };
  const cs: Pt[] = [];
  for (let k = 0; k < 4; k++) { const c = meet(lines[k], lines[(k + 1) % 4]); if (!c) return null; cs.push([Math.min(w - 1, Math.max(0, c[0])), Math.min(h - 1, Math.max(0, c[1]))]); }
  const by = (f: (p: Pt) => number) => cs.reduce((m, p) => (f(p) < f(m) ? p : m));
  const q: Quad = [by((p) => p[0] + p[1]), by((p) => -(p[0] - p[1])), by((p) => -(p[0] + p[1])), by((p) => p[0] - p[1])];
  return new Set(q).size === 4 ? q : null;
}

/**
 * Find the page: the largest bright region that the dark background cannot
 * reach from the image border. Returns its four corners in full-resolution
 * pixels, or null when the page already fills the frame or nothing page-like
 * is found (then the image is used as it is).
 */
export function detectPage(g: GrayImage): Quad | null {
  const { img: small, scale } = shrinkMean(g, 360);
  const b = boxBlur(small, 2);
  const { width: w, height: h } = b;
  const thr = otsu(b);
  // background = not-bright pixels connected to the border
  const bg = new Uint8Array(w * h);
  const stack: number[] = [];
  const push = (i: number) => { if (!bg[i] && b.data[i] <= thr) { bg[i] = 1; stack.push(i); } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (stack.length) {
    const i = stack.pop()!, x = i % w, y = (i / w) | 0;
    if (x > 0) push(i - 1);
    if (x < w - 1) push(i + 1);
    if (y > 0) push(i - w);
    if (y < h - 1) push(i + w);
  }
  let bgCount = 0;
  for (let i = 0; i < bg.length; i++) bgCount += bg[i];
  if (bgCount < w * h * 0.04) return null; // the page fills the frame already
  // largest connected component of what is left
  const lab = new Int32Array(w * h);
  let best = 0, bestN = 0, cur = 0;
  for (let s = 0; s < w * h; s++) {
    if (bg[s] || lab[s]) continue;
    cur++;
    let n = 0;
    stack.push(s);
    lab[s] = cur;
    while (stack.length) {
      const i = stack.pop()!, x = i % w, y = (i / w) | 0;
      n++;
      for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) if (j >= 0 && !bg[j] && !lab[j]) { lab[j] = cur; stack.push(j); }
    }
    if (n > bestN) { bestN = n; best = cur; }
  }
  if (bestN < w * h * 0.12) return null;
  // corners = extreme points along the two diagonals
  let tl: Pt = [0, 0], tr: Pt = [0, 0], br: Pt = [0, 0], bl: Pt = [0, 0];
  let mTl = Infinity, mTr = -Infinity, mBr = -Infinity, mBl = Infinity;
  const edge: Pt[] = [];
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (lab[y * w + x] !== best) continue;
      const s = x + y, d = x - y;
      if (s < mTl) { mTl = s; tl = [x, y]; }
      if (s > mBr) { mBr = s; br = [x, y]; }
      if (d > mTr) { mTr = d; tr = [x, y]; }
      if (d < mBl) { mBl = d; bl = [x, y]; }
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || lab[y * w + x - 1] !== best || lab[y * w + x + 1] !== best || lab[(y - 1) * w + x] !== best || lab[(y + 1) * w + x] !== best) edge.push([x, y]);
    }
  // Something dark near a corner of the page (a portrait photo, a stamp, a thumb) lets the background
  // eat into it, and the extreme point then sits well inside the page. The four straight sides are
  // still there, so the corners are taken as the crossings of the sides instead.
  const q: Quad = sidesQuad(edge, w, h) ?? [tl, tr, br, bl];
  const area = quadArea(q);
  // must look like a quadrilateral: the region fills the quad, and the quad is a real crop
  if (area < w * h * 0.12 || bestN / area < 0.7 || bestN / area > 1.2 || area > w * h * 0.96) return null;
  return q.map(([x, y]) => [x / scale, y / scale]) as Quad;
}

/** Solve the 3x3 homography that maps the unit-rectangle corners (dst) onto the quad (src). */
function homography(dst: Pt[], src: Pt[]): number[] {
  const A: number[][] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = dst[i], [u, v] = src[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  for (let c = 0; c < 8; c++) {
    let p = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < 8; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 9; k++) A[r][k] -= f * A[c][k];
    }
  }
  return [...A.map((r, i) => r[8] / r[i]), 1];
}

/** Perspective-warp the quad to an upright rectangle (bilinear sampling). */
export function warpQuad(rgba: Uint8ClampedArray, w: number, h: number, q: Quad): { data: Uint8ClampedArray; width: number; height: number } {
  const d = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const W = Math.round(Math.max(d(q[0], q[1]), d(q[3], q[2])));
  const H = Math.round(Math.max(d(q[0], q[3]), d(q[1], q[2])));
  const M = homography([[0, 0], [W, 0], [W, H], [0, H]], q);
  const out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const z = M[6] * x + M[7] * y + 1;
      const sx = Math.min(w - 1.001, Math.max(0, (M[0] * x + M[1] * y + M[2]) / z));
      const sy = Math.min(h - 1.001, Math.max(0, (M[3] * x + M[4] * y + M[5]) / z));
      const x0 = sx | 0, y0 = sy | 0, fx = sx - x0, fy = sy - y0;
      const i = (y0 * w + x0) * 4, o = (y * W + x) * 4;
      for (let c = 0; c < 3; c++)
        out[o + c] = rgba[i + c] * (1 - fx) * (1 - fy) + rgba[i + 4 + c] * fx * (1 - fy) + rgba[i + w * 4 + c] * (1 - fx) * fy + rgba[i + w * 4 + 4 + c] * fx * fy;
      out[o + 3] = 255;
    }
  return { data: out, width: W, height: H };
}

/**
 * Even out the lighting: estimate the paper brightness around every pixel
 * (local maximum, then smoothed) and divide by it, so shadows and dark corners
 * become white while ink stays dark. Works in place on RGBA.
 */
export function flattenLighting(rgba: Uint8ClampedArray, w: number, h: number): void {
  const B = Math.max(8, Math.round(Math.max(w, h) / 90));
  const gw = Math.ceil(w / B), gh = Math.ceil(h / B);
  let grid = new Float32Array(gw * gh);
  for (let gy = 0; gy < gh; gy++)
    for (let gx = 0; gx < gw; gx++) {
      // 90th percentile of brightness in the block ~ paper colour
      const hist = new Uint16Array(256);
      let n = 0;
      for (let y = gy * B; y < Math.min(h, gy * B + B); y += 2)
        for (let x = gx * B; x < Math.min(w, gx * B + B); x += 2) {
          const i = (y * w + x) * 4;
          hist[(rgba[i] * 299 + rgba[i + 1] * 587 + rgba[i + 2] * 114) / 1000 | 0]++;
          n++;
        }
      let acc = 0, v = 255;
      for (; v > 0; v--) { acc += hist[v]; if (acc >= n * 0.1) break; }
      grid[gy * gw + gx] = Math.max(40, v);
    }
  // dilate (text-heavy blocks borrow the paper level of their neighbours), then blur
  for (const pass of ['max', 'avg', 'avg'] as const) {
    const o = new Float32Array(gw * gh);
    for (let y = 0; y < gh; y++)
      for (let x = 0; x < gw; x++) {
        let m = 0, s = 0, n = 0;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= gw || yy >= gh) continue;
            const v = grid[yy * gw + xx];
            m = Math.max(m, v); s += v; n++;
          }
        o[y * gw + x] = pass === 'max' ? m : s / n;
      }
    grid = o;
  }
  for (let y = 0; y < h; y++) {
    const gy = Math.min(gh - 1.001, Math.max(0, y / B - 0.5)), y0 = gy | 0, fy = gy - y0;
    for (let x = 0; x < w; x++) {
      const gx = Math.min(gw - 1.001, Math.max(0, x / B - 0.5)), x0 = gx | 0, fx = gx - x0;
      const g0 = y0 * gw + x0;
      const bgv = grid[g0] * (1 - fx) * (1 - fy) + grid[g0 + 1] * fx * (1 - fy) + grid[g0 + gw] * (1 - fx) * fy + grid[g0 + gw + 1] * fx * fy;
      const k = 255 / bgv, i = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        // divide by the paper level, then a gentle contrast stretch
        const t = (rgba[i + c] * k) / 255;
        rgba[i + c] = 255 * Math.min(1, Math.max(0, (t - 0.08) / 0.86));
      }
    }
  }
}
