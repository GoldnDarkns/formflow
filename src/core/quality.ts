// Is this photo good enough to read? Two things ruin a phone photo of a
// document: blur (hand shake, out of focus) and glare (a lamp reflected off
// a laminated card). Both are measured from the pixels alone, before any
// reading, so the app can say "take it again" instead of showing wrong text.
//
// Sharpness: on a sharp page the edges of letters are abrupt, so the second
// difference (how fast the brightness slope changes) is large compared with the
// slope itself. Blur spreads every edge out and that ratio falls. Using a ratio
// taken only at the strongest edges makes the number independent of how much
// text the page has.
//
// Glare: a patch that is pure white with no texture at all, on a page whose
// paper is otherwise clearly darker than pure white.
import type { GrayImage } from './types';
import { resize } from './preprocess';

export interface PhotoQuality {
  sharpness: number; // about 0.2 (very blurred) to 1.5 (crisp)
  glare: number; // share of the page covered by burnt-out patches, 0..1
  paper: number; // typical brightness of the page, 0..255
  blurry: boolean;
  glary: boolean;
}

/** Below this the letters are too smeared to trust (set from the made-up specimens: see scripts/eval-quality.ts). */
export const BLUR_BELOW = 0.3;
/** At or above this share of burnt-out page, part of the text is probably hidden. */
export const GLARE_ABOVE = 0.012;
const SIDE = 900;

export function photoQuality(img: GrayImage): PhotoQuality {
  const k = SIDE / Math.max(img.width, img.height);
  const g = k < 0.98 ? resize(img, k) : img;
  const { width: W, height: H, data: d } = g;
  // slope and second difference at every inner pixel
  const n = (W - 2) * (H - 2);
  const grad = new Float32Array(n), lap = new Float32Array(n);
  let i = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++, i++) {
      const p = y * W + x;
      const gx = d[p + 1] - d[p - 1], gy = d[p + W] - d[p - W];
      grad[i] = Math.abs(gx) + Math.abs(gy);
      lap[i] = Math.abs(d[p - 1] + d[p + 1] + d[p - W] + d[p + W] - 4 * d[p]);
    }
  }
  // the strongest 4% of edges
  const hist = new Uint32Array(1024);
  for (let j = 0; j < n; j++) hist[Math.min(1023, grad[j] | 0)]++;
  let need = Math.max(50, Math.round(n * 0.04)), cut = 1023;
  for (; cut > 8 && need > 0; cut--) need -= hist[cut];
  let sg = 0, sl = 0;
  for (let j = 0; j < n; j++) if (grad[j] > cut) { sg += grad[j]; sl += lap[j]; }
  const sharpness = sg > 0 ? sl / sg : 0;

  // brightness of the paper: the 70th percentile is paper on any page that is mostly background
  const bh = new Uint32Array(256);
  for (let j = 0; j < d.length; j += 3) bh[d[j]]++;
  let acc = 0, paper = 255;
  const total = Math.ceil(d.length / 3);
  for (let v = 0; v < 256; v++) { acc += bh[v]; if (acc >= total * 0.7) { paper = v; break; } }

  // burnt-out blocks: pure white and textureless
  const B = 18;
  let blocks = 0, burnt = 0;
  for (let y = 0; y + B <= H; y += B) {
    for (let x = 0; x + B <= W; x += B) {
      blocks++;
      let lo = 255, sum = 0;
      for (let yy = y; yy < y + B; yy += 2) for (let xx = x; xx < x + B; xx += 2) { const v = d[yy * W + xx]; if (v < lo) lo = v; sum += v; }
      if (lo >= 247 && sum / 81 >= 252) burnt++;
    }
  }
  const glare = paper < 248 && blocks ? burnt / blocks : 0; // a scan on pure white paper is not glare
  return { sharpness, glare, paper, blurry: sharpness < BLUR_BELOW, glary: glare >= GLARE_ABOVE };
}
