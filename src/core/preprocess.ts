// Classical document pre-processing (Week 2 / Lecture F):
// grayscale -> deskew -> adaptive binarisation (Sauvola) -> ruled-line removal.
// Everything works on plain typed arrays so it runs in a Web Worker or in Node.

import type { GrayImage } from './types';

export function rgbaToGray(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): GrayImage {
  const data = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < data.length; i++, j += 4) {
    data[i] = (rgba[j] * 299 + rgba[j + 1] * 587 + rgba[j + 2] * 114) / 1000;
  }
  return { width, height, data };
}

export function grayToRgba(g: GrayImage): Uint8ClampedArray {
  const out = new Uint8ClampedArray(g.width * g.height * 4);
  for (let i = 0, j = 0; i < g.data.length; i++, j += 4) {
    out[j] = out[j + 1] = out[j + 2] = g.data[i];
    out[j + 3] = 255;
  }
  return out;
}

/** Nearest-neighbour downscale so the longest side is at most `max` px. */
export function downscale(g: GrayImage, max: number): { img: GrayImage; scale: number } {
  const s = Math.min(1, max / Math.max(g.width, g.height));
  if (s === 1) return { img: g, scale: 1 };
  const w = Math.max(1, Math.round(g.width * s));
  const h = Math.max(1, Math.round(g.height * s));
  const data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(g.height - 1, Math.floor(y / s));
    for (let x = 0; x < w; x++) data[y * w + x] = g.data[sy * g.width + Math.min(g.width - 1, Math.floor(x / s))];
  }
  return { img: { width: w, height: h, data }, scale: s };
}

/** Bilinear resize by a factor (used to upscale low-resolution scans for OCR). */
export function resize(g: GrayImage, f: number): GrayImage {
  if (f === 1) return g;
  const w = Math.max(1, Math.round(g.width * f)), h = Math.max(1, Math.round(g.height * f));
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(g.height - 1.001, Math.max(0, (y + 0.5) / f - 0.5));
    const y0 = Math.floor(sy), fy = sy - y0;
    for (let x = 0; x < w; x++) {
      const sx = Math.min(g.width - 1.001, Math.max(0, (x + 0.5) / f - 0.5));
      const x0 = Math.floor(sx), fx = sx - x0, i = y0 * g.width + x0;
      out[y * w + x] =
        g.data[i] * (1 - fx) * (1 - fy) + g.data[i + 1] * fx * (1 - fy) +
        g.data[i + g.width] * (1 - fx) * fy + g.data[i + g.width + 1] * fx * fy;
    }
  }
  return { width: w, height: h, data: out };
}

/** Global Otsu threshold. */
export function otsu(g: GrayImage): number {
  const hist = new Float64Array(256);
  for (const v of g.data) hist[v]++;
  const total = g.data.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, best = 0, thr = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB) continue;
    const wF = total - wB;
    if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) ** 2;
    if (between > best) { best = between; thr = t; }
  }
  return thr;
}

/**
 * Sauvola adaptive binarisation using integral images. Handles uneven lighting
 * from phone photos far better than a single global threshold.
 */
export function sauvola(g: GrayImage, window = 0, k = 0.2): GrayImage {
  const { width: w, height: h, data } = g;
  const win = window || Math.max(15, Math.round(Math.min(w, h) / 40) | 1);
  const r = win >> 1;
  const W = w + 1;
  const integ = new Float64Array(W * (h + 1));
  const integSq = new Float64Array(W * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0, rowSq = 0;
    for (let x = 0; x < w; x++) {
      const v = data[y * w + x];
      row += v; rowSq += v * v;
      integ[(y + 1) * W + x + 1] = integ[y * W + x + 1] + row;
      integSq[(y + 1) * W + x + 1] = integSq[y * W + x + 1] + rowSq;
    }
  }
  const out = new Uint8Array(w * h);
  const R = 128;
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
      const n = (x1 - x0 + 1) * (y1 - y0 + 1);
      const a = (y1 + 1) * W + x1 + 1, b = y0 * W + x1 + 1, c = (y1 + 1) * W + x0, d = y0 * W + x0;
      const s = integ[a] - integ[b] - integ[c] + integ[d];
      const sq = integSq[a] - integSq[b] - integSq[c] + integSq[d];
      const mean = s / n;
      const sd = Math.sqrt(Math.max(0, sq / n - mean * mean));
      const t = mean * (1 + k * (sd / R - 1));
      out[y * w + x] = data[y * w + x] > t ? 255 : 0;
    }
  }
  return { width: w, height: h, data: out };
}

/**
 * Estimate skew angle (degrees) by maximising the variance of the horizontal
 * projection profile of dark pixels over candidate angles.
 */
export function estimateSkew(g: GrayImage, maxDeg = 6, step = 0.25): number {
  const { img } = downscale(g, 900);
  const thr = otsu(img);
  const pts: number[] = [];
  for (let y = 0; y < img.height; y += 1)
    for (let x = 0; x < img.width; x += 2) if (img.data[y * img.width + x] < thr) pts.push(x, y);
  if (pts.length < 200) return 0;
  const H = img.height + img.width;
  const prof = new Float64Array(H * 2);
  const score = (deg: number) => {
    const t = (deg * Math.PI) / 180, s = Math.sin(t), c = Math.cos(t);
    prof.fill(0);
    for (let i = 0; i < pts.length; i += 2) {
      const yy = Math.round(pts[i + 1] * c - pts[i] * s) + H;
      prof[yy]++;
    }
    let sum = 0, sq = 0;
    for (const v of prof) { sum += v; sq += v * v; }
    return sq - (sum * sum) / prof.length;
  };
  let best = 0, bestScore = -Infinity;
  for (let d = -maxDeg; d <= maxDeg + 1e-9; d += step) {
    const sc = score(d);
    if (sc > bestScore) { bestScore = sc; best = d; }
  }
  // refine
  for (let d = best - step; d <= best + step; d += step / 5) {
    const sc = score(d);
    if (sc > bestScore) { bestScore = sc; best = d; }
  }
  return Math.abs(best) < 0.15 ? 0 : best;
}

/** Rotate around the centre by `deg` (bilinear, white background). */
export function rotate(g: GrayImage, deg: number): GrayImage {
  if (!deg) return g;
  const { width: w, height: h, data } = g;
  const t = (deg * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
  const cx = w / 2, cy = h / 2;
  const out = new Uint8Array(w * h).fill(255);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx, dy = y - cy;
      const sx = c * dx - s * dy + cx, sy = s * dx + c * dy + cy;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1) continue;
      const fx = sx - x0, fy = sy - y0, i = y0 * w + x0;
      out[y * w + x] =
        data[i] * (1 - fx) * (1 - fy) + data[i + 1] * fx * (1 - fy) + data[i + w] * (1 - fx) * fy + data[i + w + 1] * fx * fy;
    }
  }
  return { width: w, height: h, data: out };
}

/**
 * Remove long thin horizontal and vertical rules (form lines, table borders),
 * which otherwise break OCR and grouping. Works on a binary image.
 */
export function removeLines(b: GrayImage, minFrac = 0.06, maxThick = 0): GrayImage {
  const { width: w, height: h } = b;
  const out = new Uint8Array(b.data);
  const thick = maxThick || Math.max(2, Math.round(Math.min(w, h) / 400));
  const minH = Math.round(w * minFrac), minV = Math.round(h * minFrac);
  const mask = new Uint8Array(w * h);
  // horizontal runs
  for (let y = 0; y < h; y++) {
    let run = 0;
    for (let x = 0; x <= w; x++) {
      const dark = x < w && b.data[y * w + x] === 0;
      if (dark) run++;
      else {
        if (run >= minH) for (let k = x - run; k < x; k++) mask[y * w + k] = 1;
        run = 0;
      }
    }
  }
  // vertical runs
  for (let x = 0; x < w; x++) {
    let run = 0;
    for (let y = 0; y <= h; y++) {
      const dark = y < h && b.data[y * w + x] === 0;
      if (dark) run++;
      else {
        if (run >= minV) for (let k = y - run; k < y; k++) mask[k * w + x] = 2;
        run = 0;
      }
    }
  }
  // erase masked pixels only where the stroke is thin perpendicular to the line
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const m = mask[y * w + x];
      if (!m) continue;
      if (m === 1) {
        let up = 0, dn = 0;
        while (up <= thick * 2 && y - up - 1 >= 0 && b.data[(y - up - 1) * w + x] === 0) up++;
        while (dn <= thick * 2 && y + dn + 1 < h && b.data[(y + dn + 1) * w + x] === 0) dn++;
        if (up + dn + 1 <= thick * 2 + 1) out[y * w + x] = 255;
      } else {
        let l = 0, r = 0;
        while (l <= thick * 2 && x - l - 1 >= 0 && b.data[y * w + x - l - 1] === 0) l++;
        while (r <= thick * 2 && x + r + 1 < w && b.data[y * w + x + r + 1] === 0) r++;
        if (l + r + 1 <= thick * 2 + 1) out[y * w + x] = 255;
      }
    }
  }
  return { width: w, height: h, data: out };
}

/** Upscale factor so small scans get ~2000 px on the long side (Tesseract likes ~30 px text). */
export function ocrScale(w: number, h: number): number {
  const long = Math.max(w, h);
  if (long >= 1800) return 1;
  return Math.min(2.5, 2000 / long);
}

/** Single global Otsu threshold (best for clean scans and faxes). */
export function otsuBinarise(g: GrayImage): GrayImage {
  const t = otsu(g);
  const out = new Uint8Array(g.data.length);
  for (let i = 0; i < out.length; i++) out[i] = g.data[i] > t ? 255 : 0;
  return { width: g.width, height: g.height, data: out };
}

/**
 * How uneven is the paper background? Std-dev of the bright (paper) level across
 * 64 px blocks. Phone photos with shadows score high; flat scans score low.
 */
export function backgroundUnevenness(g: GrayImage): number {
  const B = 64, levels: number[] = [];
  for (let by = 0; by + B <= g.height; by += B)
    for (let bx = 0; bx + B <= g.width; bx += B) {
      let mx = 0;
      const hist = new Uint32Array(256);
      for (let y = by; y < by + B; y += 2) for (let x = bx; x < bx + B; x += 2) hist[g.data[y * g.width + x]]++;
      // 90th percentile = paper brightness in this block
      let acc = 0, n = (B / 2) * (B / 2);
      for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= n * 0.9) { mx = v; break; } }
      levels.push(mx);
    }
  if (levels.length < 4) return 0;
  const m = levels.reduce((a, b) => a + b, 0) / levels.length;
  return Math.sqrt(levels.reduce((a, b) => a + (b - m) ** 2, 0) / levels.length);
}

/** Otsu for evenly lit scans, Sauvola (local) when the background varies. */
export function binariseAuto(g: GrayImage): { img: GrayImage; method: 'otsu' | 'sauvola' } {
  return backgroundUnevenness(g) > 12 ? { img: sauvola(g), method: 'sauvola' } : { img: otsuBinarise(g), method: 'otsu' };
}

export interface PreprocessOptions {
  deskew: boolean;
  binarise: boolean;
  removeLines: boolean;
  magic?: boolean; // find the page, flatten it and fix the lighting first (see scan.ts)
  reader?: 'tesseract' | 'ppocr'; // which text reader the app uses
}

export const DEFAULT_PRE: PreprocessOptions = { deskew: true, binarise: true, removeLines: true, magic: true };

export function preprocess(g: GrayImage, opt: PreprocessOptions = DEFAULT_PRE): { img: GrayImage; skew: number } {
  let img = g;
  let skew = 0;
  if (opt.deskew) {
    skew = estimateSkew(img);
    img = rotate(img, skew);
  }
  if (opt.binarise) img = binariseAuto(img).img;
  if (opt.binarise && opt.removeLines) img = removeLines(img);
  return { img, skew };
}
