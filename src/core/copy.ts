// The digital copy of a document: everything that was read, laid out where it was printed.
// Words are grouped into lines (and a line is split where a wide gap separates two columns), the
// Arabic segments and the code lines are added, and each line is linked to the key fact or field it
// carries, so a correction made on the copy reaches the summary and the exports.
import type { Box, OcrWord } from './types';
import type { Fact } from './facts';
import { parseDate, plainArabic } from './facts';
import type { MrzResult } from './mrz';
import { rowsOf } from './classify';
import { editDistance } from './facts';

export interface Piece {
  id: string; // stable for one document: made from the position
  text: string;
  box: Box;
  conf: number; // 0..1
  kind: 'text' | 'arabic' | 'mrz' | 'mark';
  /** the key fact or field this line carries */
  fact?: string;
  field?: string; // field uid
  /** the part of the line that is the fact's value, as printed */
  valueText?: string;
  verified?: boolean; // proven by a check digit or checksum
}
export interface FieldLike { uid: string; key: string; value: string; confidence: number; box?: Box | null; mark?: boolean }
export interface ArabicLike { text: string; box: Box; conf: number }

const overlap = (a: Box, b: Box) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
const area = (b: Box) => Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
const union = (bs: Box[]): Box => [Math.min(...bs.map((b) => b[0])), Math.min(...bs.map((b) => b[1])), Math.max(...bs.map((b) => b[2])), Math.max(...bs.map((b) => b[3]))];
const flat = (s: string) => s.toUpperCase().replace(/[^A-Z0-9]/g, '');
const idOf = (b: Box) => `p${Math.round(b[0])}-${Math.round(b[1])}`;

/** A word that is only noise: one or two marks of punctuation read with little confidence. */
const noise = (w: OcrWord) => !/[\p{L}\p{N}]/u.test(w.text) && w.conf < 60;

export function piecesOf(words: OcrWord[], arabic: ArabicLike[], facts: Fact[], mrz: MrzResult | null, fields: FieldLike[] = []): Piece[] {
  // Latin words lying on an Arabic segment are the English reader's attempt at Arabic: drop them
  const ws = words.filter((w) => !noise(w) && !arabic.some((a) => overlap(w.box, a.box) > 0.5 * area(w.box)));
  const out: Piece[] = [];
  for (const row of rowsOf(ws)) {
    const h = row.reduce((s, w) => s + (w.box[3] - w.box[1]), 0) / row.length;
    let cur: OcrWord[] = [];
    const flush = () => {
      if (!cur.length) return;
      const box = union(cur.map((w) => w.box));
      out.push({ id: idOf(box), text: cur.map((w) => w.text).join(' '), box, conf: cur.reduce((s, w) => s + w.conf, 0) / cur.length / 100, kind: 'text' });
      cur = [];
    };
    for (const w of row) {
      // a gap wider than about one and a half letters' height separates two columns (a label and a value printed far apart stay apart)
      if (cur.length && w.box[0] - cur[cur.length - 1].box[2] > h * 1.6) flush();
      cur.push(w);
    }
    flush();
  }
  for (const a of arabic) out.push({ id: idOf(a.box), text: a.text, box: a.box, conf: a.conf, kind: 'arabic' });

  // the code lines: rows full of "<" are replaced by the code line as parsed, checked by its digits
  if (mrz?.lines.length) {
    const rows = out.filter((p) => p.kind === 'text' && (p.text.match(/</g)?.length ?? 0) >= 3);
    rows.sort((a, b) => a.box[1] - b.box[1]);
    const used = new Set<Piece>();
    // pieces on the same row as a code line belong to it (the reader may have split it)
    const lines = mrz.lines.map((line, i) => {
      const r = rows[rows.length - mrz.lines.length + i];
      if (!r) return null;
      const same = out.filter((p) => p.kind === 'text' && Math.abs((p.box[1] + p.box[3]) / 2 - (r.box[1] + r.box[3]) / 2) < (r.box[3] - r.box[1]) * 0.6);
      same.forEach((p) => used.add(p));
      const box = union(same.map((p) => p.box));
      return { id: idOf(box), text: line, box, conf: mrz.valid ? 1 : 0.7, kind: 'mrz' as const, verified: mrz.valid };
    });
    if (lines.every(Boolean)) {
      for (let i = out.length - 1; i >= 0; i--) if (used.has(out[i])) out.splice(i, 1);
      out.push(...(lines as Piece[]));
    }
  }

  // link each key fact to the line that prints it
  for (const f of facts) {
    if (!f.value) continue;
    // a value a person corrected is found by what was read, so the line keeps its link
    const p = findValue(out, f.read ? { ...f, value: f.read } : f);
    if (p && !p.fact) { p.fact = f.key; p.verified ||= !!(f.proof && f.valid); }
  }
  // and each field (on forms: the answer box) to the line that overlaps it most
  for (const fl of fields) {
    if (!fl.box) continue;
    let best: Piece | null = null, bo = 0;
    for (const p of out) { const o = overlap(p.box, fl.box) / Math.min(area(p.box), area(fl.box)); if (o > bo) { bo = o; best = p; } }
    if (fl.mark) { out.push({ id: idOf(fl.box) + 'm', text: /^(Ticked|Signed)$/.test(fl.value) ? '☑' : '☐', box: fl.box, conf: fl.confidence, kind: 'mark', field: fl.uid }); continue; }
    if (best && bo > 0.5 && !best.fact && !best.field) best.field = fl.uid;
  }
  return out.sort((a, b) => a.box[1] - b.box[1] || a.box[0] - b.box[0]);
}

/** The line printing a fact's value, and the part of it that is the value. */
function findValue(pieces: Piece[], f: Fact): Piece | null {
  const v = f.value;
  const cand = pieces.filter((p) => p.kind !== 'mrz');
  if (/in Arabic$/.test(f.key)) {
    const want = plainArabic(v);
    const p = cand.find((p) => p.kind === 'arabic' && plainArabic(p.text).includes(want));
    if (p) p.valueText = v;
    return p ?? null;
  }
  // a date: compare the date a line holds, whatever way it is printed
  if (/^\d{4}-\d{2}-\d{2}$/.test(v)) {
    for (const p of cand) {
      for (const m of p.text.matchAll(/\d{1,4}[\s./-]+(?:\d{1,2}|[A-Za-z]{3,9})[\s./-]+\d{2,4}/g)) if (parseDate(m[0]) === v) { p.valueText = m[0]; return p; }
    }
    return null;
  }
  const fv = flat(v);
  if (fv.length < 2) return null;
  // the value as whole words in one line ("UTO" is not inside "UTOPIA"), spacing and punctuation free
  const tokens = v.toUpperCase().split(/[^A-Z0-9]+/).filter(Boolean); // letters and digits only, nothing to escape
  const whole = new RegExp('(?<![A-Z0-9])' + tokens.join('[^A-Z0-9]*') + '(?![A-Z0-9])', 'i');
  for (const p of cand) { const m = p.text.match(whole); if (m) { p.valueText = m[0]; return p; } }
  // otherwise allowing a misread letter or two in long values
  let best: Piece | null = null, bd = Infinity;
  for (const p of cand) {
    const t = flat(p.text);
    if (fv.length >= 6) {
      for (let i = 0; i + fv.length <= t.length + 1; i++) { const d = editDistance(t.slice(i, i + fv.length), fv); if (d < bd) { bd = d; best = p; } }
    }
  }
  if (best && bd <= Math.max(1, Math.floor(fv.length / 8))) { best.valueText = v; return best; }
  // a name spread over two lines (surname, then given names): link the line holding the longest part
  const parts = v.split(/\s+/).filter((x) => x.length >= 3).sort((a, b) => b.length - a.length);
  for (const part of parts) { const p = cand.find((p) => flat(p.text).includes(flat(part))); if (p) return p; }
  return null;
}

/** What a line says after the person's corrections. */
export interface Correction { text?: string; verdict: 'right' | 'wrong' }
export const correctedText = (p: Piece, c?: Correction) => (c?.text !== undefined ? c.text : p.text);
