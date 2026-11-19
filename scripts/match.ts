// Field matching shared by the evaluator and the calibration script.
import type { Field } from '../src/core/types';
import { cx, cy } from '../src/core/layout';
import { norm, similarity } from '../src/core/ocr';
import type { GtPage } from './funsd';

export function gtFields(gt: GtPage) {
  const byId = new Map(gt.ents.map((e) => [e.id, e]));
  const m = new Map<number, number[]>();
  for (const k of gt.qa) {
    const [q, a] = k.split(',').map(Number);
    if (!m.has(q)) m.set(q, []);
    m.get(q)!.push(a);
  }
  return [...m].map(([q, as]) => ({
    key: byId.get(q)!.text.replace(/[:\s.]+$/, ''),
    value: as.map((a) => byId.get(a)!).sort((x, y) => cy(x.box) - cy(y.box) || cx(x.box) - cx(y.box)).map((e) => e.text).join(' '),
  }));
}
export const clean = (s: string) => norm(s).replace(/[^\p{L}\p{N} ]/gu, '').replace(/\s+/g, ' ').trim();
export const sim = (a: string, b: string) => similarity(clean(a), clean(b));

/** A predicted field is right when both key and value are at least `thr` similar to an unused gold field. */
export function matchFields(pred: Field[], gold: { key: string; value: string }[], thr = 0.8) {
  const P = pred.filter((f) => f.value);
  const used = new Set<number>();
  const res = P.map((f) => {
    let best = -1, bs = 0;
    gold.forEach((g, j) => {
      if (used.has(j)) return;
      const ks = sim(f.key, g.key), vs = sim(f.value, g.value);
      if (ks >= thr && vs >= thr && ks + vs > bs) { bs = ks + vs; best = j; }
    });
    if (best >= 0) used.add(best);
    const exact = best >= 0 && clean(f.value) === clean(gold[best].value);
    return { f, ok: best >= 0, exact };
  });
  return { res, tp: res.filter((r) => r.ok).length, nPred: P.length, nGold: gold.length };
}
