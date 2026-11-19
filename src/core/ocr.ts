// Helpers to turn a Tesseract.js result into flat word lists, and to score OCR
// against ground truth (character error rate, hallucination, omission).

import type { Box, OcrWord } from './types';
import { iou, coverage } from './layout';

interface TBbox { x0: number; y0: number; x1: number; y1: number }
interface TWord { text: string; confidence: number; bbox: TBbox }
interface TLine { words: TWord[] }
interface TPara { lines: TLine[] }
interface TBlock { paragraphs: TPara[] }

export function wordsFromTesseract(blocks: TBlock[] | null, scale = 1, minConf = 0): OcrWord[] {
  const out: OcrWord[] = [];
  let line = 0;
  for (const b of blocks ?? [])
    for (const p of b.paragraphs)
      for (const l of p.lines) {
        for (const w of l.words) {
          const text = w.text.trim();
          if (!text || w.confidence < minConf) continue;
          out.push({
            text,
            conf: w.confidence,
            line,
            box: [w.bbox.x0 / scale, w.bbox.y0 / scale, w.bbox.x1 / scale, w.bbox.y1 / scale],
          });
        }
        line++;
      }
  return out;
}

/** Strip form artefacts that Tesseract reads as characters (underscores, rule dashes, bars). */
export function cleanWords(words: OcrWord[]): OcrWord[] {
  const out: OcrWord[] = [];
  for (const w of words) {
    const text = w.text
      .replace(/_+/g, '')
      .replace(/^[—–\-|=~]+(?=\S)|(?<=\S)[—–\-|=~]+$/g, '')
      .replace(/^[|]+$/, '')
      .trim();
    if (!text || /^[—–\-=~_|.,'`]+$/.test(text)) continue;
    out.push({ ...w, text });
  }
  return out;
}

export function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  let prev = new Array(b.length + 1).fill(0).map((_, i) => i);
  let cur = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    [prev, cur] = [cur, prev];
  }
  return prev[b.length];
}

export const norm = (s: string) => s.toLowerCase().replace(/\s+/g, ' ').trim();

/** character similarity in 0..1 */
export function similarity(a: string, b: string): number {
  const A = norm(a), B = norm(b);
  if (!A && !B) return 1;
  return 1 - levenshtein(A, B) / Math.max(A.length, B.length);
}

/**
 * Best approximate occurrence of `needle` inside `hay` (Sellers algorithm):
 * returns the minimum edit distance of needle against any substring of hay.
 */
export function substringDistance(needle: string, hay: string): number {
  const n = needle.length, m = hay.length;
  if (!n) return 0;
  let prev = new Array(m + 1).fill(0);
  let cur = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    cur[0] = i;
    for (let j = 1; j <= m; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (needle[i - 1] === hay[j - 1] ? 0 : 1));
    }
    [prev, cur] = [cur, prev];
  }
  return Math.min(...prev);
}

export interface GtWord { text: string; box: Box }

export interface OcrScore {
  gtChars: number;
  errors: number;
  cer: number;
  gtWords: number;
  omitted: number; // GT words with no overlapping prediction
  predWords: number;
  hallucinated: number; // predicted words not overlapping any GT word
  omissionRate: number;
  hallucinationRate: number;
  wordAcc: number;
}

/**
 * Word-level matching by box overlap. Each GT word is compared to the
 * concatenation of predicted words that mostly lie inside it (handles
 * split/merged words); unmatched GT words count as fully wrong.
 */
export function scoreOcr(gt: GtWord[], pred: { text: string; box: Box }[]): OcrScore {
  const gtW = gt.filter((w) => w.text.trim());
  const used = new Set<number>();
  let errors = 0, chars = 0, omitted = 0, correct = 0;
  for (const g of gtW) {
    const t = norm(g.text);
    chars += t.length;
    const hits: number[] = [];
    pred.forEach((p, i) => {
      if (coverage(p.box, g.box) > 0.5 || iou(p.box, g.box) > 0.3) hits.push(i);
    });
    if (!hits.length) {
      omitted++;
      errors += t.length;
      continue;
    }
    hits.forEach((i) => used.add(i));
    hits.sort((a, b) => pred[a].box[0] - pred[b].box[0]);
    const pt = norm(hits.map((i) => pred[i].text).join(' '));
    // a prediction may span several GT words; look for the GT word inside it
    const d = pt.length > t.length + 2 ? substringDistance(t, pt) : levenshtein(t, pt);
    errors += Math.min(t.length, d);
    if (d === 0) correct++;
  }
  let hallucinated = 0;
  pred.forEach((p, i) => {
    if (used.has(i)) return;
    if (!gtW.some((g) => coverage(p.box, g.box) > 0.2 || coverage(g.box, p.box) > 0.2)) hallucinated++;
  });
  return {
    gtChars: chars,
    errors,
    cer: chars ? errors / chars : 0,
    gtWords: gtW.length,
    omitted,
    predWords: pred.length,
    hallucinated,
    omissionRate: gtW.length ? omitted / gtW.length : 0,
    hallucinationRate: pred.length ? hallucinated / pred.length : 0,
    wordAcc: gtW.length ? correct / gtW.length : 0,
  };
}
