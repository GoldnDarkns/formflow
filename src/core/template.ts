// Template mode for known forms (the brief's stretch goal).
//
// A template is taught once from one copy of a form: the printed words and
// where they are (anchors), plus a labelled box for every place a value goes
// (slots). A new page is recognised by finding the same printed words, the two
// are aligned with a least-squares affine fit, and each slot is read by taking
// the words that fall inside it. No model is involved, so it cannot guess a
// label wrongly: the label comes from the person who taught the form.
import type { Box, Entity, Field, Link, OcrWord } from './types';

export interface Anchor { t: string; x: number; y: number; w: number; h: number } // normalised 0..1, centre + size
export interface Slot { label: string; box: Box } // normalised 0..1
export interface Template {
  id: string;
  name: string;
  source?: string; // where the blank form came from
  filled?: boolean; // taught from a filled-in copy rather than a blank form
  aspect: number; // width / height of the taught page
  anchors: Anchor[];
  slots: Slot[];
  checks?: Slot[]; // checkboxes: label + box, read by ink instead of text
}

/** a, b, c, d, e, f with  x' = a x + b y + c,  y' = d x + e y + f  (template -> page, normalised) */
export type Affine = [number, number, number, number, number, number];
export interface TemplateMatch { template: Template; affine: Affine; inliers: number; score: number }

const key = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
const inside = (x: number, y: number, b: Box, pad = 0) => x >= b[0] - pad && x <= b[2] + pad && y >= b[1] - pad && y <= b[3] + pad;

/** Teach a template from one read page. Words inside a slot are treated as filled-in values, not print. */
export function buildTemplate(id: string, name: string, words: OcrWord[], W: number, H: number, slots: Slot[], source?: string, checks?: Slot[]): Template {
  const anchors: Anchor[] = [];
  for (const w of words) {
    const t = key(w.text);
    if (t.length < 2) continue;
    const x = (w.box[0] + w.box[2]) / 2 / W, y = (w.box[1] + w.box[3]) / 2 / H;
    if (slots.some((s) => inside(x, y, s.box))) continue;
    anchors.push({ t, x: r4(x), y: r4(y), w: r4((w.box[2] - w.box[0]) / W), h: r4((w.box[3] - w.box[1]) / H) });
  }
  const t: Template = { id, name, source, aspect: W / H, anchors, slots: slots.map((s) => ({ label: s.label, box: s.box.map(r4) as Box })) };
  if (checks?.length) t.checks = checks.map((s) => ({ label: s.label, box: s.box.map(r4) as Box }));
  return t;
}
const r4 = (v: number) => Math.round(v * 1e4) / 1e4;

/** Least-squares affine from point pairs (solves two 3x3 normal systems). */
function fitAffine(p: { x: number; y: number; u: number; v: number }[]): Affine | null {
  let sxx = 0, sxy = 0, sx = 0, syy = 0, sy = 0, su = 0, sxu = 0, syu = 0, sv = 0, sxv = 0, syv = 0;
  const n = p.length;
  if (n < 3) return null;
  for (const q of p) {
    sxx += q.x * q.x; sxy += q.x * q.y; sx += q.x; syy += q.y * q.y; sy += q.y;
    su += q.u; sxu += q.x * q.u; syu += q.y * q.u; sv += q.v; sxv += q.x * q.v; syv += q.y * q.v;
  }
  const det = sxx * (syy * n - sy * sy) - sxy * (sxy * n - sy * sx) + sx * (sxy * sy - syy * sx);
  if (Math.abs(det) < 1e-12) return null;
  const solve = (r0: number, r1: number, r2: number): [number, number, number] => [
    (r0 * (syy * n - sy * sy) - sxy * (r1 * n - sy * r2) + sx * (r1 * sy - syy * r2)) / det,
    (sxx * (r1 * n - r2 * sy) - r0 * (sxy * n - sy * sx) + sx * (sxy * r2 - r1 * sx)) / det,
    (sxx * (syy * r2 - sy * r1) - sxy * (sxy * r2 - r1 * sx) + r0 * (sxy * sy - syy * sx)) / det,
  ];
  return [...solve(sxu, syu, su), ...solve(sxv, syv, sv)];
}
export const applyAffine = (m: Affine, x: number, y: number): [number, number] => [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];
const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };

/** Does this page look like the template? Aligns them and counts printed words found in place. */
export function matchOne(t: Template, words: OcrWord[], W: number, H: number): TemplateMatch | null {
  const page = new Map<string, { x: number; y: number }[]>();
  for (const w of words) {
    const k = key(w.text);
    if (k.length < 2) continue;
    if (!page.has(k)) page.set(k, []);
    page.get(k)!.push({ x: (w.box[0] + w.box[2]) / 2 / W, y: (w.box[1] + w.box[3]) / 2 / H });
  }
  // start from words that occur once on both pages: an unambiguous correspondence
  const count = new Map<string, number>();
  for (const a of t.anchors) count.set(a.t, (count.get(a.t) ?? 0) + 1);
  let pairs: { x: number; y: number; u: number; v: number }[] = [];
  for (const a of t.anchors) {
    const c = page.get(a.t);
    if (a.t.length >= 3 && count.get(a.t) === 1 && c?.length === 1) pairs.push({ x: a.x, y: a.y, u: c[0].x, v: c[0].y });
  }
  if (pairs.length < 6) return null;
  // drop pairs far from the typical shift, fit, then keep what the fit explains and refit
  const dx = median(pairs.map((p) => p.u - p.x)), dy = median(pairs.map((p) => p.v - p.y));
  pairs = pairs.filter((p) => Math.abs(p.u - p.x - dx) < 0.08 && Math.abs(p.v - p.y - dy) < 0.08);
  let m = fitAffine(pairs);
  if (!m) return null;
  for (let it = 0; it < 2; it++) {
    const keep = pairs.filter((p) => { const [u, v] = applyAffine(m!, p.x, p.y); return Math.hypot(u - p.u, v - p.v) < 0.012; });
    const m2 = keep.length >= 6 ? fitAffine(keep) : null;
    if (!m2) break;
    m = m2;
  }
  // a sane fit is close to "same size, hardly rotated"
  if (m[0] < 0.75 || m[0] > 1.3 || m[4] < 0.75 || m[4] > 1.3 || Math.abs(m[1]) > 0.12 || Math.abs(m[3]) > 0.12) return null;
  // score: share of all printed words that are found where the fit says they should be
  let inliers = 0;
  for (const a of t.anchors) {
    const [u, v] = applyAffine(m, a.x, a.y);
    if (page.get(a.t)?.some((c) => Math.abs(c.x - u) < 0.012 + a.w / 2 && Math.abs(c.y - v) < 0.008 + a.h / 2)) inliers++;
  }
  return { template: t, affine: m, inliers, score: inliers / Math.max(1, t.anchors.length) };
}

export const MIN_SCORE = 0.45;
export const MIN_INLIERS = 15;

/** The best-matching template, if any is convincing. */
export function matchTemplate(templates: Template[], words: OcrWord[], W: number, H: number): TemplateMatch | null {
  let best: TemplateMatch | null = null;
  for (const t of templates) {
    const m = matchOne(t, words, W, H);
    if (m && (!best || m.inliers > best.inliers)) best = m;
  }
  return best && best.score >= MIN_SCORE && best.inliers >= MIN_INLIERS ? best : null;
}

/** Where the template's checkboxes are on this page, in page pixels. */
export function templateChecks(match: TemplateMatch, W: number, H: number): { label: string; box: Box }[] {
  return (match.template.checks ?? []).map((s) => {
    const [x0, y0] = applyAffine(match.affine, s.box[0], s.box[1]);
    const [x1, y1] = applyAffine(match.affine, s.box[2], s.box[3]);
    return { label: s.label, box: [Math.min(x0, x1) * W, Math.min(y0, y1) * H, Math.max(x0, x1) * W, Math.max(y0, y1) * H] as Box };
  });
}

/** Where the printed words that pass `test` are on this page (page pixels), e.g. every "Signature" caption. */
export function templateWords(match: TemplateMatch, W: number, H: number, test: (t: string) => boolean): Box[] {
  return match.template.anchors.filter((a) => test(a.t)).map((a) => {
    const [x, y] = applyAffine(match.affine, a.x, a.y);
    return [(x - a.w / 2) * W, (y - a.h / 2) * H, (x + a.w / 2) * W, (y + a.h / 2) * H] as Box;
  });
}

export interface TemplateRead { entities: Entity[]; links: Link[]; fields: Field[] }

/** Read every slot of a matched template from the page's words. */
export function readTemplate(match: TemplateMatch, words: OcrWord[], W: number, H: number): TemplateRead {
  const { template: t, affine: m } = match;
  // where each printed word should be on this page, to tell print from filled-in text
  const printed = t.anchors.map((a) => { const [u, v] = applyAffine(m, a.x, a.y); return { t: a.t, x: u * W, y: v * H, w: a.w * W, h: a.h * H }; });
  const isPrint = (w: OcrWord) => {
    const k = key(w.text);
    if (!k) return true;
    const x = (w.box[0] + w.box[2]) / 2, y = (w.box[1] + w.box[3]) / 2;
    return printed.some((p) => p.t === k && Math.abs(p.x - x) < 0.012 * W + p.w / 2 && Math.abs(p.y - y) < 0.008 * H + p.h / 2);
  };
  const entities: Entity[] = [];
  const links: Link[] = [];
  const fields: Field[] = [];
  const used = new Set<OcrWord>();
  for (const s of t.slots) {
    const c = [[s.box[0], s.box[1]], [s.box[2], s.box[1]], [s.box[2], s.box[3]], [s.box[0], s.box[3]]].map(([x, y]) => applyAffine(m, x, y));
    const box: Box = [Math.min(...c.map((p) => p[0])) * W, Math.min(...c.map((p) => p[1])) * H, Math.max(...c.map((p) => p[0])) * W, Math.max(...c.map((p) => p[1])) * H];
    const padY = Math.max(2, (box[3] - box[1]) * 0.15);
    const labelKeys = new Set(s.label.split(/\s+/).map(key).filter((k) => k.length >= 3));
    const grown: Box = [box[0], box[1] - padY, box[2], box[3] + padY];
    const inSlot = words
      .filter((w) => !used.has(w) && inside((w.box[0] + w.box[2]) / 2, (w.box[1] + w.box[3]) / 2, grown) && !isPrint(w))
      .sort((a, b) => (Math.abs(a.box[1] - b.box[1]) > (a.box[3] - a.box[1]) * 0.6 ? a.box[1] - b.box[1] : a.box[0] - b.box[0]));
    // a template taught from a filled-in copy may have a box that includes its printed label: drop those leading words
    const vals = t.filled && inSlot.length > 1 ? inSlot.filter((w, i) => !(i < labelKeys.size && labelKeys.has(key(w.text)))) : inSlot;
    vals.forEach((w) => used.add(w));
    const qId = entities.length;
    entities.push({ id: qId, text: s.label, box: [box[0], box[1], box[0] + 1, box[1] + 1], words: [], label: 'question', probs: [1, 0, 0, 0] });
    const value = vals.map((w) => w.text).join(' ');
    const ocr = vals.length ? Math.min(...vals.map((w) => w.conf)) / 100 : 0;
    const aId = entities.length;
    entities.push({ id: aId, text: value, box, words: vals, label: 'answer', probs: [0, 1, 0, 0] });
    links.push({ q: qId, a: aId, p: match.score });
    const reasons: string[] = [];
    if (value && ocr < 0.7) reasons.push('Low OCR confidence');
    fields.push({
      key: s.label, value, qId, aId,
      confidence: value ? ocr : 0.5,
      raw: ocr,
      status: !value ? 'empty' : reasons.length ? 'review' : 'ok',
      reasons: value ? reasons : ['Nothing written in this box'],
      source: 'template',
    });
  }
  return { entities, links, fields };
}
