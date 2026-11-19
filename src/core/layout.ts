// Layout analysis: group OCR words into entities (text segments), compute
// features for the entity-label and question->answer link classifiers, and
// decode the classifier outputs into key-value pairs.

import type { Box, Entity, Label, Link, OcrWord } from './types';
import { LABELS } from './types';

// ---------- geometry ----------
export const bw = (b: Box) => b[2] - b[0];
export const bh = (b: Box) => b[3] - b[1];
export const cx = (b: Box) => (b[0] + b[2]) / 2;
export const cy = (b: Box) => (b[1] + b[3]) / 2;
export function union(boxes: Box[]): Box {
  return [
    Math.min(...boxes.map((b) => b[0])),
    Math.min(...boxes.map((b) => b[1])),
    Math.max(...boxes.map((b) => b[2])),
    Math.max(...boxes.map((b) => b[3])),
  ];
}
export function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const u = bw(a) * bh(a) + bw(b) * bh(b) - inter;
  return u > 0 ? inter / u : 0;
}
/** fraction of a's area covered by b */
export function coverage(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const ar = bw(a) * bh(a);
  return ar > 0 ? (ix * iy) / ar : 0;
}
export function vOverlap(a: Box, b: Box): number {
  const o = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return o / Math.max(1, Math.min(bh(a), bh(b)));
}
export function hOverlap(a: Box, b: Box): number {
  const o = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  return o / Math.max(1, Math.min(bw(a), bw(b)));
}
export function edgeDist(a: Box, b: Box): number {
  const dx = Math.max(0, Math.max(a[0], b[0]) - Math.min(a[2], b[2]));
  const dy = Math.max(0, Math.max(a[1], b[1]) - Math.min(a[3], b[3]));
  return Math.hypot(dx, dy);
}
export function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
}

// ---------- grouping ----------
export interface GroupOptions {
  gapFactor: number; // split a line where the gap exceeds gapFactor * line height
  splitOnColon: boolean;
  mergeLines: boolean; // merge left-aligned continuation lines into one entity
}
export const DEFAULT_GROUP: GroupOptions = { gapFactor: 1.1, splitOnColon: true, mergeLines: true };

const isNoise = (w: OcrWord) => !/[\p{L}\p{N}]/u.test(w.text) && w.text.length < 2 && !/[:/&#%$+]/.test(w.text);

export function groupWords(words: OcrWord[], opt: GroupOptions = DEFAULT_GROUP): Entity[] {
  const byLine = new Map<number, OcrWord[]>();
  for (const w of words) {
    if (!w.text.trim() || isNoise(w)) continue;
    if (!byLine.has(w.line)) byLine.set(w.line, []);
    byLine.get(w.line)!.push(w);
  }
  const segs: OcrWord[][] = [];
  for (const lw of byLine.values()) {
    lw.sort((a, b) => a.box[0] - b.box[0]);
    const h = median(lw.map((w) => bh(w.box))) || 10;
    let cur: OcrWord[] = [];
    for (const w of lw) {
      const prev = cur[cur.length - 1];
      if (prev) {
        const gap = w.box[0] - prev.box[2];
        const colonBreak = opt.splitOnColon && /:$/.test(prev.text);
        if (gap > opt.gapFactor * h || colonBreak) {
          segs.push(cur);
          cur = [];
        }
      }
      cur.push(w);
    }
    if (cur.length) segs.push(cur);
  }
  let ents: Entity[] = segs.map((ws, i) => ({
    id: i,
    words: ws,
    text: ws.map((w) => w.text).join(' '),
    box: union(ws.map((w) => w.box)),
  }));
  if (opt.mergeLines) ents = mergeContinuations(ents);
  ents.sort((a, b) => a.box[1] - b.box[1] || a.box[0] - b.box[0]);
  return ents.map((e, i) => ({ ...e, id: i }));
}

/** Merge a segment with the one directly below it when it is a wrapped continuation. */
function mergeContinuations(ents: Entity[]): Entity[] {
  const sorted = [...ents].sort((a, b) => a.box[1] - b.box[1]);
  const used = new Set<number>();
  const out: Entity[] = [];
  for (let i = 0; i < sorted.length; i++) {
    if (used.has(i)) continue;
    let e = sorted[i];
    for (;;) {
      const h = bh(e.box) / Math.max(1, new Set(e.words.map((w) => w.line)).size);
      if (/:$/.test(e.text)) break;
      let best = -1;
      for (let j = i + 1; j < sorted.length; j++) {
        if (used.has(j)) continue;
        const f = sorted[j];
        if (f.box[1] - e.box[3] > 0.7 * h) break;
        const gap = f.box[1] - e.box[3];
        if (gap < -0.3 * h) continue;
        const aligned = Math.abs(f.box[0] - e.box[0]) < 0.8 * h;
        // compare typical word heights: a small label above a large value is not a continuation
        const he = median(e.words.map((w) => bh(w.box))), hf = median(f.words.map((w) => bh(w.box)));
        const sameSize = hf / Math.max(1, he) < 1.2 && he / Math.max(1, hf) < 1.2;
        // "Surname / Nom" above "ERIKSSON": mixed-case label over an ALL-CAPS value
        const caps = (t: string) => /\p{L}{2}/u.test(t) && t === t.toUpperCase();
        if (caps(f.text) !== caps(e.text)) continue;
        if (/\s\/\s/.test(e.text)) continue; // bilingual label lines stand alone
        const startsLower = /^[a-z]/.test(f.text);
        const tall = sorted.some((g, k) => k !== i && k !== j && !used.has(k) && vOverlap(g.box, f.box) > 0.5 && g.box[2] < f.box[0] && f.box[0] - g.box[2] < 3 * h);
        if (aligned && sameSize && !/:/.test(f.text) && !tall && (startsLower || bw(f.box) < bw(e.box) * 1.3)) {
          best = j;
          break;
        }
      }
      if (best < 0) break;
      const f = sorted[best];
      used.add(best);
      e = { id: e.id, words: [...e.words, ...f.words], text: e.text + ' ' + f.text, box: union([e.box, f.box]) };
    }
    out.push(e);
  }
  return out;
}

// ---------- features ----------
const HASH_BUCKETS = 48;
function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % HASH_BUCKETS;
}

function textStats(t: string) {
  const letters = t.match(/\p{L}/gu)?.length ?? 0;
  const upper = t.match(/\p{Lu}/gu)?.length ?? 0;
  const digits = t.match(/\d/g)?.length ?? 0;
  const punct = t.match(/[^\p{L}\p{N}\s]/gu)?.length ?? 0;
  const n = Math.max(1, t.replace(/\s/g, '').length);
  const words = t.split(/\s+/).filter(Boolean);
  const title = words.filter((w) => /^\p{Lu}\p{Ll}/u.test(w)).length / Math.max(1, words.length);
  return {
    nWords: words.length,
    nChars: n,
    fracUpper: letters ? upper / letters : 0,
    fracDigit: digits / n,
    fracAlpha: letters / n,
    fracPunct: punct / n,
    allCaps: letters >= 2 && upper === letters ? 1 : 0,
    title,
    endsColon: /:\s*$/.test(t) ? 1 : 0,
    hasColon: /:/.test(t) ? 1 : 0,
    endsQ: /\?\s*$/.test(t) ? 1 : 0,
    endsDigit: /\d\s*$/.test(t) ? 1 : 0,
    startsDigit: /^\s*\d/.test(t) ? 1 : 0,
    hasDate: /\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b(19|20)\d{2}\b/.test(t) ? 1 : 0,
  };
}

export interface PageCtx {
  W: number;
  H: number;
  lh: number; // median word height on the page
}
export function pageCtx(ents: Entity[], W: number, H: number): PageCtx {
  const hs = ents.flatMap((e) => (e.words.length ? e.words.map((w) => bh(w.box)) : [bh(e.box)]));
  return { W, H, lh: median(hs) || 12 };
}

const cap = (v: number, m = 50) => Math.max(-m, Math.min(m, v));

export function entityFeatures(ents: Entity[], ctx: PageCtx): number[][] {
  const { W, H, lh } = ctx;
  return ents.map((e) => {
    const s = textStats(e.text);
    const lines = new Set(e.words.map((w) => w.line)).size || 1;
    const conf = e.words.length ? e.words.reduce((a, w) => a + w.conf, 0) / e.words.length / 100 : 1;
    let right: Entity | null = null, left: Entity | null = null, below: Entity | null = null, above: Entity | null = null;
    let dR = 1e9, dL = 1e9, dB = 1e9, dA = 1e9;
    let nSameLine = 0;
    for (const f of ents) {
      if (f === e) continue;
      if (vOverlap(e.box, f.box) > 0.5) {
        nSameLine++;
        const g = f.box[0] - e.box[2];
        if (g >= -lh * 0.3 && g < dR) { dR = g; right = f; }
        const gl = e.box[0] - f.box[2];
        if (gl >= -lh * 0.3 && gl < dL) { dL = gl; left = f; }
      }
      if (hOverlap(e.box, f.box) > 0.2) {
        const gb = f.box[1] - e.box[3];
        if (gb >= -lh * 0.3 && gb < dB) { dB = gb; below = f; }
        const ga = e.box[1] - f.box[3];
        if (ga >= -lh * 0.3 && ga < dA) { dA = ga; above = f; }
      }
    }
    const tokens = new Array(HASH_BUCKETS).fill(0);
    for (const t of e.text.toLowerCase().match(/[a-z]{2,}/g) ?? []) tokens[hash(t)] = 1;
    const f = [
      Math.log1p(s.nWords), Math.log1p(s.nChars), s.fracUpper, s.fracDigit, s.fracAlpha, s.fracPunct,
      s.allCaps, s.title, s.endsColon, s.hasColon, s.endsQ, s.endsDigit, s.startsDigit, s.hasDate,
      e.box[0] / W, e.box[1] / H, cx(e.box) / W, bw(e.box) / W, bh(e.box) / lines / lh, lines, conf,
      right ? 1 : 0, right ? cap(dR / lh) : 50, right ? textStats(right.text).endsColon : 0, right ? textStats(right.text).fracDigit : 0,
      left ? 1 : 0, left ? cap(dL / lh) : 50, left ? textStats(left.text).endsColon : 0,
      below ? 1 : 0, below ? cap(dB / lh) : 50,
      above ? 1 : 0, above ? cap(dA / lh) : 50, above ? textStats(above.text).endsColon : 0,
      nSameLine,
      ...tokens,
    ];
    return f;
  });
}

export interface Pair {
  q: number;
  a: number;
}

export function candidatePairs(ents: Entity[], qIdx: number[], aIdx: number[], ctx: PageCtx): Pair[] {
  const out: Pair[] = [];
  for (const q of qIdx)
    for (const a of aIdx) {
      if (q === a) continue;
      const Q = ents[q].box, A = ents[a].box;
      if (Math.abs(cy(A) - cy(Q)) > 25 * ctx.lh) continue;
      if (edgeDist(Q, A) > 40 * ctx.lh) continue;
      out.push({ q, a });
    }
  return out;
}

export function pairFeatures(ents: Entity[], pairs: Pair[], ctx: PageCtx): number[][] {
  const { lh } = ctx;
  // rank answers for each question and questions for each answer by edge distance
  const dist = new Map<string, number>();
  const byQ = new Map<number, number[]>(), byA = new Map<number, number[]>();
  for (const p of pairs) {
    const d = edgeDist(ents[p.q].box, ents[p.a].box);
    dist.set(p.q + ',' + p.a, d);
    if (!byQ.has(p.q)) byQ.set(p.q, []);
    if (!byA.has(p.a)) byA.set(p.a, []);
    byQ.get(p.q)!.push(d);
    byA.get(p.a)!.push(d);
  }
  for (const v of byQ.values()) v.sort((a, b) => a - b);
  for (const v of byA.values()) v.sort((a, b) => a - b);
  return pairs.map(({ q, a }) => {
    const Q = ents[q], A = ents[a];
    const qb = Q.box, ab = A.box;
    const d = dist.get(q + ',' + a)!;
    const rq = byQ.get(q)!.indexOf(d), ra = byA.get(a)!.indexOf(d);
    const qs = textStats(Q.text), as = textStats(A.text);
    const U = union([qb, ab]);
    let between = 0;
    for (const e of ents) {
      if (e === Q || e === A) continue;
      if (cx(e.box) > U[0] && cx(e.box) < U[2] && cy(e.box) > U[1] && cy(e.box) < U[3]) between++;
    }
    const vo = vOverlap(qb, ab), ho = hOverlap(qb, ab);
    return [
      cap((ab[0] - qb[2]) / lh), cap((ab[0] - qb[0]) / lh), cap((ab[1] - qb[1]) / lh), cap((ab[1] - qb[3]) / lh),
      cap((cy(ab) - cy(qb)) / lh), cap((cx(ab) - cx(qb)) / lh), cap(d / lh), vo, ho,
      vo > 0.5 ? 1 : 0, ab[0] >= qb[2] - lh * 0.5 ? 1 : 0, ab[1] >= qb[3] - lh * 0.3 ? 1 : 0,
      Math.min(rq, 10), Math.min(ra, 10), Math.min(between, 10),
      qs.endsColon, Math.log1p(qs.nWords), Math.log1p(as.nWords), as.fracDigit, qs.fracUpper, as.endsColon,
    ];
  });
}

// ---------- decoding ----------
export function argmaxLabel(probs: number[]): Label {
  let k = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[k]) k = i;
  return LABELS[k];
}

/** each answer links to at most one question (the most probable one above threshold) */
export function decodeLinks(pairs: Pair[], probs: number[], thr = 0.5): Link[] {
  const best = new Map<number, Link>();
  pairs.forEach((p, i) => {
    const pr = probs[i];
    if (pr < thr) return;
    const cur = best.get(p.a);
    if (!cur || pr > cur.p) best.set(p.a, { q: p.q, a: p.a, p: pr });
  });
  return [...best.values()];
}
