// Marks that are not text: ticked checkboxes and pen signatures.
// Classical computer vision on the cleaned grey page, no model.
import type { Box, GrayImage, OcrWord } from './types';
import { otsu } from './preprocess';
import { levenshtein } from './ocr';

export interface Checkbox { box: Box; checked: boolean; ink: number }
export interface SignatureMark { caption: Box; box: Box | null; signed: boolean; ink: number }

/** dark = 1. One global threshold, nudged up a little so thin box borders survive a soft scan. */
function darkMask(g: GrayImage): Uint8Array {
  const thr = Math.min(200, otsu(g) + 10);
  const m = new Uint8Array(g.width * g.height);
  for (let i = 0; i < m.length; i++) m[i] = g.data[i] < thr ? 1 : 0;
  return m;
}

interface Comp { x0: number; y0: number; x1: number; y1: number; n: number }
/** 8-connected components of a mask inside a region; `visit` gets every pixel index of each component. */
function components(m: Uint8Array, W: number, H: number, region?: Box): Comp[] {
  const rx0 = Math.max(0, Math.floor(region?.[0] ?? 0)), ry0 = Math.max(0, Math.floor(region?.[1] ?? 0));
  const rx1 = Math.min(W - 1, Math.ceil(region?.[2] ?? W - 1)), ry1 = Math.min(H - 1, Math.ceil(region?.[3] ?? H - 1));
  const seen = new Uint8Array(W * H);
  const out: Comp[] = [];
  const stack: number[] = [];
  for (let y = ry0; y <= ry1; y++) {
    for (let x = rx0; x <= rx1; x++) {
      const s = y * W + x;
      if (!m[s] || seen[s]) continue;
      const c: Comp = { x0: x, y0: y, x1: x, y1: y, n: 0 };
      stack.push(s);
      seen[s] = 1;
      while (stack.length) {
        const i = stack.pop()!;
        const px = i % W, py = (i - px) / W;
        c.n++;
        if (px < c.x0) c.x0 = px;
        if (px > c.x1) c.x1 = px;
        if (py < c.y0) c.y0 = py;
        if (py > c.y1) c.y1 = py;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = py + dy;
          if (ny < ry0 || ny > ry1) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = px + dx;
            if (nx < rx0 || nx > rx1) continue;
            const j = ny * W + nx;
            if (m[j] && !seen[j]) { seen[j] = 1; stack.push(j); }
          }
        }
      }
      out.push(c);
    }
  }
  return out;
}

/** share of dark pixels in a rectangle of the mask */
function inkIn(m: Uint8Array, W: number, H: number, b: Box): number {
  const x0 = Math.max(0, Math.round(b[0])), y0 = Math.max(0, Math.round(b[1])), x1 = Math.min(W, Math.round(b[2])), y1 = Math.min(H, Math.round(b[3]));
  if (x1 <= x0 || y1 <= y0) return 0;
  let n = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) n += m[y * W + x];
  return n / ((x1 - x0) * (y1 - y0));
}

export const CHECKED_INK = 0.1;

/** Is the box at this place ticked? Looks at the middle of the box only, so the border does not count. */
export function checkState(g: GrayImage, box: Box, mask?: Uint8Array): { checked: boolean; ink: number } {
  const m = mask ?? darkMask(g);
  const w = box[2] - box[0], h = box[3] - box[1];
  const ix = Math.max(2, w * 0.24), iy = Math.max(2, h * 0.24);
  const ink = inkIn(m, g.width, g.height, [box[0] + ix, box[1] + iy, box[2] - ix, box[3] - iy]);
  return { checked: ink > CHECKED_INK, ink };
}

/**
 * Find checkboxes: small, nearly square outlines. A tick or cross inside joins
 * the outline, so the outline is tested on its four sides, not on its filling.
 */
export function findCheckboxes(g: GrayImage): Checkbox[] {
  const W = g.width, H = g.height;
  const m = darkMask(g);
  const minS = Math.max(8, W * 0.0085), maxS = W * 0.03;
  const out: Checkbox[] = [];
  for (const c of components(m, W, H)) {
    const w = c.x1 - c.x0 + 1, h = c.y1 - c.y0 + 1;
    if (w < minS || h < minS || w > maxS || h > maxS || w / h < 0.78 || w / h > 1.28) continue;
    // each side: most of a thin band along the edge is dark somewhere across its thickness
    const t = Math.max(2, Math.round(Math.min(w, h) * 0.14));
    const side = (horizontal: boolean, start: number) => {
      const len = horizontal ? w : h;
      let hit = 0;
      for (let i = 0; i < len; i++) {
        let d = 0;
        for (let k = 0; k < t && !d; k++) d = horizontal ? m[(start + k) * W + c.x0 + i] : m[(c.y0 + i) * W + start + k];
        hit += d;
      }
      return hit / len;
    };
    const sides = [side(true, c.y0), side(true, c.y1 - t + 1), side(false, c.x0), side(false, c.x1 - t + 1)];
    if (Math.min(...sides) < 0.8) continue;
    // a drawn box has ink in all four corners; joined letters such as "IS" rarely do
    const cs = Math.max(2, Math.round(Math.min(w, h) * 0.16));
    const corners = [[c.x0, c.y0], [c.x1 + 1 - cs, c.y0], [c.x0, c.y1 + 1 - cs], [c.x1 + 1 - cs, c.y1 + 1 - cs]].map(([x, y]) => inkIn(m, W, H, [x, y, x + cs, y + cs]));
    if (Math.min(...corners) < 0.45) continue;
    // a box is drawn with a thin line; a letter such as D or B has a thick stroke
    const run = (horizontal: boolean, at: number) => {
      let n = 0;
      const len = horizontal ? w : h;
      while (n < len && (horizontal ? m[at * W + c.x0 + n] : m[(c.y0 + n) * W + at])) n++;
      return n;
    };
    const thickL = Math.min(run(true, c.y0 + Math.round(h * 0.35)), run(true, c.y0 + Math.round(h * 0.5)), run(true, c.y0 + Math.round(h * 0.65)));
    const thickT = Math.min(run(false, c.x0 + Math.round(w * 0.35)), run(false, c.x0 + Math.round(w * 0.5)), run(false, c.x0 + Math.round(w * 0.65)));
    if (thickL > Math.max(3, w * 0.2) || thickT > Math.max(3, h * 0.2)) continue;
    // an outline, not a solid block
    if (c.n / (w * h) > 0.8) continue;
    const box: Box = [c.x0, c.y0, c.x1 + 1, c.y1 + 1];
    out.push({ box, ...checkState(g, box, m) });
  }
  return out.sort((a, b) => a.box[1] - b.box[1] || a.box[0] - b.box[0]);
}

/**
 * Drop squares that are really letters. The reader often glues a true checkbox to the
 * caption after it, so a square at the very start of a "word" is kept; a square further
 * inside a word is a letter such as D or O. `scale` = image pixels per word-coordinate unit.
 */
export function dropLetterBoxes(found: Checkbox[], words: OcrWord[], scale = 1): Checkbox[] {
  return found.filter((f) => {
    const x0 = f.box[0] / scale, x = (f.box[0] + f.box[2]) / 2 / scale, y = (f.box[1] + f.box[3]) / 2 / scale, bw = (f.box[2] - f.box[0]) / scale;
    return !words.some((w) => /[A-Za-z0-9]{2,}/.test(w.text) && x > w.box[0] && x < w.box[2] && y > w.box[1] && y < w.box[3] && x0 > w.box[0] + bw * 0.5);
  });
}

/**
 * Signatures: for each printed "Signature..." caption, look around the signing line
 * (above the caption and to its right) for pen strokes. A pen stroke is a connected
 * mark much wider than a printed letter and mostly empty inside its outline; ruled
 * lines and box frames are removed first by deleting long straight runs.
 */
export const isSignatureCaption = (t: string) => { const k = t.replace(/[^A-Za-z0-9]/g, '').toLowerCase().slice(0, 9); return k.length >= 8 && levenshtein(k, 'signature') <= 2; };

export function findSignatures(g: GrayImage, words: OcrWord[], scale = 1, captions?: Box[]): SignatureMark[] {
  const W = g.width, H = g.height;
  // the caption word, allowing a couple of misread letters ("Signatura", "5ignature")
  const isCaption = isSignatureCaption;
  // a template knows where the captions are even when the reader missed them on this copy
  const caps: { box: Box }[] = captions?.length ? captions.map((box) => ({ box })) : words.filter((w) => isCaption(w.text));
  if (!caps.length) return [];
  const m = darkMask(g);
  const heights = words.map((w) => (w.box[3] - w.box[1]) * scale).sort((a, b) => a - b);
  const th = heights[heights.length >> 1] || 16; // typical text height
  // delete long straight runs (ruled lines, box frames) so only curved pen strokes and letters remain
  const pen = m.slice();
  // (upright pen strokes can be as tall as a box side, so only very long vertical runs go;
  // what is left of a frame is a thin upright bar, which is too narrow to pass as a stroke)
  const maxH = Math.round(th * 3), maxV = Math.round(th * 6);
  for (let y = 0; y < H; y++) {
    let run = 0;
    for (let x = 0; x <= W; x++) {
      if (x < W && m[y * W + x]) run++;
      else { if (run >= maxH) for (let k = x - run; k < x; k++) pen[y * W + k] = 0; run = 0; }
    }
  }
  for (let x = 0; x < W; x++) {
    let run = 0;
    for (let y = 0; y <= H; y++) {
      if (y < H && m[y * W + x]) run++;
      else { if (run >= maxV) for (let k = y - run; k < y; k++) pen[k * W + x] = 0; run = 0; }
    }
  }
  // printed and typed words are not signatures: blank them out (a scribble is rarely read as a confident word)
  for (const w of words) {
    if (w.conf < 60 || !/[A-Za-z0-9]{2,}/.test(w.text)) continue;
    const x0 = Math.max(0, Math.floor(w.box[0] * scale) - 1), x1 = Math.min(W, Math.ceil(w.box[2] * scale) + 1);
    const y0 = Math.max(0, Math.floor(w.box[1] * scale) - 1), y1 = Math.min(H, Math.ceil(w.box[3] * scale) + 1);
    for (let y = y0; y < y1; y++) pen.fill(0, y * W + x0, y * W + x1);
  }
  const out: SignatureMark[] = [];
  for (const c of caps) {
    const b = c.box.map((v) => v * scale) as Box;
    const h = b[3] - b[1];
    if (out.some((o) => Math.abs(o.caption[1] - c.box[1]) < (h / scale) * 2 && Math.abs(o.caption[0] - c.box[0]) < (W / scale) * 0.16)) continue;
    const region: Box = [b[0] - W * 0.01, b[1] - Math.max(h * 4.2, th * 2.6), b[0] + W * 0.42, b[3] + h * 0.6];
    const strokes = components(pen, W, H, region).filter((s) => {
      const sw = s.x1 - s.x0 + 1, sh = s.y1 - s.y0 + 1;
      return sw >= th * 2.2 && sh >= th * 0.7 && s.n / (sw * sh) < 0.4 && s.n >= th * 3;
    });
    let box: Box | null = null, ink = 0;
    for (const s of strokes) {
      box = box ? [Math.min(box[0], s.x0), Math.min(box[1], s.y0), Math.max(box[2], s.x1 + 1), Math.max(box[3], s.y1 + 1)] : [s.x0, s.y0, s.x1 + 1, s.y1 + 1];
      ink += s.n;
    }
    out.push({ caption: c.box, box: box ? (box.map((v) => v / scale) as Box) : null, signed: !!box, ink });
  }
  return out;
}

/** The caption printed just right of a checkbox, on the same line (stops at a wide gap or the next box). */
export function checkboxLabel(box: Box, words: OcrWord[], others: Box[] = []): string {
  const h = box[3] - box[1], cy = (box[1] + box[3]) / 2;
  const line = words.filter((w) => Math.abs((w.box[1] + w.box[3]) / 2 - cy) < h * 0.7 && w.box[0] >= box[2] - h * 0.3).sort((a, b) => a.box[0] - b.box[0]);
  const stop = Math.min(...others.filter((o) => o[0] > box[2] && Math.abs((o[1] + o[3]) / 2 - cy) < h * 0.7).map((o) => o[0]), Infinity);
  const out: string[] = [];
  let end = box[2];
  for (const w of line) {
    if (w.box[0] >= stop || w.box[0] - end > h * 2.2 || out.length >= 9) break;
    out.push(w.text);
    end = w.box[2];
  }
  return out.join(' ').replace(/^[^A-Za-z0-9(]+/, '').trim();
}
