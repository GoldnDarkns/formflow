// Learned corner finder: DocCornerNet LEAN (MIT, 1.8 MB, from the scanic project). A small network
// (MobileNetV2 backbone) looks at the whole picture squeezed to 224 x 224 and says where the four
// corners of a document are, plus how likely it is that a document is there at all. It does not
// depend on the background being plain, which is where the classical finders in scan.ts fail
// (stripes, grids, wood grain, white on white). Its corners are rough (a few pixels at 224), so they
// are sharpened afterwards on the full picture with refineQuad, exactly like the classical ones.
import type { Quad } from './scan';

export const CORNER_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

interface OrtLike {
  InferenceSession: { create(src: any, opts?: any): Promise<any> };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => any;
}
export interface CornerResult { quad: Quad; score: number }
export interface CornerModel {
  /** `rgba` is the picture already squeezed to 224 x 224; corners come back as fractions (0..1) of the picture */
  find(rgba: Uint8ClampedArray | Uint8Array): Promise<CornerResult>;
}

/** 224 x 224 RGBA -> the model's input: channels last, ImageNet normalisation */
export function cornerInput(rgba: Uint8ClampedArray | Uint8Array): Float32Array {
  const n = CORNER_SIZE * CORNER_SIZE, t = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) t[i * 3 + c] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  return t;
}

export async function loadCornerModel(ort: OrtLike, src: any, sessionOpts?: any): Promise<CornerModel> {
  const s = await ort.InferenceSession.create(src, sessionOpts);
  return {
    async find(rgba) {
      const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', cornerInput(rgba), [1, CORNER_SIZE, CORNER_SIZE, 3]) });
      let coords: Float32Array | null = null, logit = 0;
      for (const k of s.outputNames) { const d = out[k].data as Float32Array; if (d.length === 8) coords = d; else if (d.length === 1) logit = d[0]; }
      if (!coords) throw new Error('corner model gave no corners');
      const c = coords;
      // order: top left, top right, bottom right, bottom left
      return { quad: [0, 1, 2, 3].map((i) => [Math.min(1, Math.max(0, c[i * 2])), Math.min(1, Math.max(0, c[i * 2 + 1]))]) as Quad, score: 1 / (1 + Math.exp(-logit)) };
    },
  };
}

/**
 * Share of points along the outline where the picture just outside differs clearly from just inside.
 * A real document edge has the table, a hand or a pattern beyond it; a box the network imagines inside
 * a page that fills the picture (a form's table, a panel) has the same paper on both sides.
 * `q` is in pixels of `rgba` (w x h).
 */
export function edgeContrast(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number, q: Quad, n = 24): number {
  const cx = q.reduce((s, p) => s + p[0], 0) / 4, cy = q.reduce((s, p) => s + p[1], 0) / 4;
  const off = Math.hypot(w, h) * 0.02;
  const at = (x: number, y: number) => { const i = (Math.min(h - 1, Math.max(0, Math.round(y))) * w + Math.min(w - 1, Math.max(0, Math.round(x)))) * 4; return [rgba[i], rgba[i + 1], rgba[i + 2]]; };
  let differ = 0, total = 0;
  for (let s = 0; s < 4; s++) {
    const a = q[s], b = q[(s + 1) % 4];
    for (let k = 1; k < n; k++) {
      const x = a[0] + ((b[0] - a[0]) * k) / n, y = a[1] + ((b[1] - a[1]) * k) / n;
      const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy) || 1;
      const ox = x + (dx / d) * off, oy = y + (dy / d) * off;
      if (ox < 0 || oy < 0 || ox >= w || oy >= h) continue; // beyond the picture: no evidence either way
      const i = at(x - (dx / d) * off, y - (dy / d) * off), o = at(ox, oy);
      total++;
      if (Math.max(Math.abs(i[0] - o[0]), Math.abs(i[1] - o[1]), Math.abs(i[2] - o[2])) > 28) differ++;
    }
  }
  return total ? differ / total : 0;
}

/** Is a quad a believable document outline: convex, not tiny, not the whole picture, sides not wildly uneven? */
export function plausibleQuad(q: Quad): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const [a, b, c] = [q[i], q[(i + 1) % 4], q[(i + 2) % 4]];
    const z = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (Math.abs(z) < 1e-9) return false;
    if (sign && Math.sign(z) !== sign) return false;
    sign = Math.sign(z);
  }
  const area = Math.abs(q.reduce((s, p, i) => s + p[0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * p[1], 0)) / 2;
  const side = (i: number) => Math.hypot(q[i][0] - q[(i + 1) % 4][0], q[i][1] - q[(i + 1) % 4][1]);
  return area > 0.04 && area < 0.97 && Math.min(side(0), side(2)) / Math.max(side(0), side(2)) > 0.5 && Math.min(side(1), side(3)) / Math.max(side(1), side(3)) > 0.5;
}
