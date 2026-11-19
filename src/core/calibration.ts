// Confidence calibration. The raw field score (a weighted blend of link
// probability, label probability and OCR certainty) ranks fields well but is
// too optimistic. scripts/calibrate.ts fits a monotone curve (isotonic
// regression) on the FUNSD *training* split that maps the raw score to the
// share of fields that were actually right; the *test* split measures the gain.
import data from './calibration.json' with { type: 'json' };

export type CurveName = 'tesseract' | 'ppocr' | 'tesseract+second';
export interface Curve { x: number[]; y: number[]; n: number }
export interface Calibration { fitted: boolean; note?: string; fittedOn?: string; curves: Partial<Record<CurveName, Curve>> }

export const CALIBRATION = data as Calibration;

/** Piecewise-linear lookup on a monotone curve. */
export function applyCurve(c: Curve, raw: number): number {
  const { x, y } = c;
  if (!x.length) return raw;
  if (raw <= x[0]) return y[0];
  for (let i = 1; i < x.length; i++) {
    if (raw <= x[i]) {
      const t = (raw - x[i - 1]) / Math.max(1e-9, x[i] - x[i - 1]);
      return y[i - 1] + t * (y[i] - y[i - 1]);
    }
  }
  return y[y.length - 1];
}

/** Raw score in 0..1 -> estimated chance the field is right. Identity when no curve is fitted. */
export function calibrate(raw: number, curve: CurveName, cal: Calibration = CALIBRATION): number {
  const r = Math.min(1, Math.max(0, raw));
  const c = cal.curves[curve] ?? cal.curves.tesseract;
  return c ? Math.min(1, Math.max(0, applyCurve(c, r))) : r;
}

/**
 * Isotonic regression (pool adjacent violators) of outcome on score, then
 * thinned to at most `knots` points. Blocks smaller than `minBlock` samples are
 * merged with a neighbour so a handful of lucky fields cannot claim 100%.
 */
export function fitIsotonic(samples: { score: number; ok: boolean }[], knots = 12, minBlock = 25): Curve {
  const s = [...samples].sort((a, b) => a.score - b.score);
  type B = { sx: number; sy: number; n: number };
  const mean = (b: B) => b.sy / b.n;
  const merge = (bs: B[], i: number) => { bs[i - 1].sx += bs[i].sx; bs[i - 1].sy += bs[i].sy; bs[i - 1].n += bs[i].n; bs.splice(i, 1); };
  // pool adjacent violators
  const blocks: B[] = [];
  for (const p of s) {
    blocks.push({ sx: p.score, sy: p.ok ? 1 : 0, n: 1 });
    while (blocks.length > 1 && mean(blocks[blocks.length - 2]) >= mean(blocks[blocks.length - 1])) merge(blocks, blocks.length - 1);
  }
  // merge under-sized blocks into a neighbour (merging neighbours keeps the curve monotone)
  for (let guard = 0; guard < 1e5; guard++) {
    const i = blocks.findIndex((b) => b.n < minBlock);
    if (i < 0 || blocks.length === 1) break;
    merge(blocks, i === 0 ? 1 : i);
  }
  let pts = blocks.map((b) => ({ x: b.sx / b.n, y: b.sy / b.n }));
  while (pts.length > knots) {
    // drop the interior point whose removal changes the curve least
    let bi = 1, bd = Infinity;
    for (let i = 1; i < pts.length - 1; i++) {
      const t = (pts[i].x - pts[i - 1].x) / Math.max(1e-9, pts[i + 1].x - pts[i - 1].x);
      const d = Math.abs(pts[i - 1].y + t * (pts[i + 1].y - pts[i - 1].y) - pts[i].y);
      if (d < bd) { bd = d; bi = i; }
    }
    pts.splice(bi, 1);
  }
  const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
  return { x: pts.map((p) => r4(p.x)), y: pts.map((p) => r4(p.y)), n: s.length };
}

/** Expected calibration error over equal-width bins, plus the bins for a reliability chart. */
export function reliability(samples: { conf: number; ok: boolean }[], nBins = 10) {
  const bins = Array.from({ length: nBins }, (_, i) => ({ lo: i / nBins, hi: (i + 1) / nBins, n: 0, conf: 0, ok: 0 }));
  for (const s of samples) {
    const b = bins[Math.min(nBins - 1, Math.floor(Math.min(1, Math.max(0, s.conf)) * nBins))];
    b.n++; b.conf += s.conf; if (s.ok) b.ok++;
  }
  const N = Math.max(1, samples.length);
  let ece = 0;
  const out = bins.map((b) => {
    const mc = b.n ? b.conf / b.n : 0, acc = b.n ? b.ok / b.n : 0;
    ece += (b.n / N) * Math.abs(mc - acc);
    return { lo: b.lo, hi: b.hi, fields: b.n, meanConfidence: mc, correct: acc };
  });
  const brier = samples.reduce((a, s) => a + (s.conf - (s.ok ? 1 : 0)) ** 2, 0) / N;
  return { ece, brier, bins: out };
}
