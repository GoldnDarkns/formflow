// Two lessons from the first test on real documents:
//
// 1. People photograph cards and booklets sideways. A page read the wrong way
//    up produces almost no clean words, so "how many clean words did we get"
//    tells us to turn the page and try again.
// 2. The code lines at the bottom of a passport are often missed by the first
//    reading pass. Their check digits make it safe to try several crops and
//    keep whichever one adds up.
import type { OcrWord } from './types';

/** Clean, confidently read words of four letters or more. A sideways page has close to none. */
export const readingQuality = (words: OcrWord[]) => words.filter((w) => /^[A-Za-z]{4,}$/.test(w.text) && w.conf >= 80).length;
/** At or below this many clean words the page is tried in other orientations. */
export const UPRIGHT_MIN = 2;
/** A turned page must beat the original by this many clean words to be kept. */
export const TURN_GAIN = 3;
export const TURNS = [90, 270, 180] as const;

/**
 * Is a code-line reading believable? Where the first pass saw rows of "<" we
 * trust that there are code lines and keep even an imperfect reading (the app
 * then shows which check digits fail). Where we only went looking at the bottom
 * of the page, at least two check digits must add up, otherwise ordinary text
 * at the foot of a form could be mistaken for code lines.
 */
export const believableMrz = (m: { checks: { ok: boolean }[]; valid: boolean; lines?: string[]; format?: string } | null, sawChevrons: boolean) =>
  !!m && (sawChevrons || m.valid || (m.checks.filter((c) => c.ok).length >= 2 && shapedLikeMrz(m.lines, m.format)));

/**
 * Do the lines have the shape of real code lines? The first line starts with the
 * document letter, and the line that carries the name holds no digits and has
 * the "<<" between surname and given names. Rows of ordinary text at the foot
 * of a vehicle licence ("EXPDATE2034...") passed two check digits by luck but
 * fail this.
 */
export function shapedLikeMrz(lines?: string[], format?: string): boolean {
  if (!lines?.length) return true; // nothing to judge by: leave the decision to the check digits
  const nameLine = format === 'TD1' ? lines[lines.length - 1] : lines[0].slice(5);
  return /^[PIACV][A-Z<]/.test(lines[0]) && !/\d/.test(nameLine) && /[A-Z]<<[A-Z]/.test(nameLine);
}

/**
 * Where the neural reader looks for the code lines: the band the first pass
 * pointed at (widened), then the bottom 45% and the bottom 60% of the page.
 * On real passports and ID cards the neural reader gets every check digit
 * right far more often than the classical one (12 of 14 against 8 of 14 on the
 * first real set; 93% against 33% on made-up specimens), so it goes first.
 */
export function mrzNeuralCrops(band: [number, number] | null, H: number, nWords: number): [number, number][] {
  if (!band && nWords >= 260) return [];
  const out: [number, number][] = [];
  if (band) out.push([Math.max(0, band[0] - (band[1] - band[0]) * 0.6), Math.min(H, band[1] + (band[1] - band[0]) * 0.3)]);
  out.push([H * 0.55, H], [H * 0.4, H]);
  return out.map(([a, b]) => [Math.round(a), Math.round(b)]);
}

/** Words from a reader, put back into lines of text (top to bottom, left to right, no spaces: code lines have none). */
export function linesFromWords(words: { text: string; box: number[] }[]): string {
  const rows: { y: number; ws: typeof words }[] = [];
  for (const w of [...words].sort((p, q) => p.box[1] - q.box[1])) {
    const cy = (w.box[1] + w.box[3]) / 2;
    const r = rows.find((r) => Math.abs(r.y - cy) < (w.box[3] - w.box[1]) * 0.6);
    if (r) r.ws.push(w); else rows.push({ y: cy, ws: [w] });
  }
  return rows.map((r) => r.ws.sort((p, q) => p.box[0] - q.box[0]).map((w) => w.text).join('')).join('\n');
}

export interface MrzCrop { y0: number; y1: number; binarised: boolean }

/**
 * Where to look for the code lines, most likely place first. `band` is where the
 * first pass saw rows of "<" (may be null); H is the image height in the same
 * pixels. Pages that are clearly long text documents with no "<" are skipped.
 */
export function mrzCrops(band: [number, number] | null, H: number, nWords: number): MrzCrop[] {
  if (!band && nWords >= 260) return [];
  const plain: [number, number][] = [];
  if (band) plain.push(band);
  plain.push([H * 0.75, H]);
  if (band) plain.push([Math.max(0, band[0] - (band[1] - band[0]) * 0.5), band[1]]);
  plain.push([H * 0.6, H]);
  const mk = (binarised: boolean) => plain.map(([y0, y1]) => ({ y0: Math.round(y0), y1: Math.round(y1), binarised }));
  return [...mk(false), ...mk(true)];
}
