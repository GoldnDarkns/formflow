// Turns OCR words into reviewed key-value fields:
// group -> label entities (classifier) -> link question->answer (classifier)
// -> fields with confidence, using agreement with the second reader if present.
import type { Entity, Field, Link, OcrWord, SecondReading } from './types';
import { LABELS } from './types';
import { argmaxLabel, candidatePairs, coverage, decodeLinks, entityFeatures, groupWords, pageCtx, pairFeatures, vOverlap, bh, hOverlap, median } from './layout';
import { norm, substringDistance } from './ocr';

export interface Classifiers {
  entity(X: number[][]): Promise<number[][]>; // probs per LABELS
  link(X: number[][]): Promise<number[]>; // P(link)
}

export interface Analysis {
  entities: Entity[];
  links: Link[];
  fields: Field[];
  agreement?: (boolean | null)[]; // per word (same order as `words`)
}

/** Does the second reader contain this word? null = it read nothing there. */
export function wordAgreement(words: OcrWord[], second: SecondReading[]): (boolean | null)[] {
  return words.map((w) => {
    const h = bh(w.box);
    const pad: [number, number, number, number] = [w.box[0] - h, w.box[1] - h * 0.6, w.box[2] + h, w.box[3] + h * 0.6];
    const regions = second.filter((r) => coverage(w.box, r.box) > 0.3 || (coverage(w.box, pad as any) > 0 && vOverlap(w.box, r.box) > 0.4 && r.box[0] <= w.box[2] + h && r.box[2] >= w.box[0] - h));
    if (!regions.length) return null;
    const t = norm(w.text).replace(/[^\p{L}\p{N}]/gu, '');
    if (!t) return true;
    const hay = regions.map((r) => norm(r.text).replace(/[^\p{L}\p{N}]/gu, '')).join('|');
    const tol = t.length >= 8 ? 1 : 0;
    return substringDistance(t, hay) <= tol;
  });
}

/** text the second reader saw inside a box (for showing an alternative value) */
export function secondText(box: Entity['box'], second: SecondReading[]): string {
  return second
    .filter((r) => coverage(r.box, box) > 0.5 || coverage(box, r.box) > 0.5)
    .sort((a, b) => a.box[1] - b.box[1] || a.box[0] - b.box[0])
    .map((r) => r.text)
    .join(' ');
}

const keyText = (t: string) => t.replace(/[:\s.]+$/, '').trim();

/**
 * ID-document layout prior (passports, ID cards, licences, visas): a label is
 * printed in small type directly above (or left of) its value in larger type.
 * Applied on top of the learned classifiers when the document is an ID.
 */
function idLayout(entities: Entity[], links: Link[]): Link[] {
  const hw = (e: Entity) => median(e.words.map((w) => bh(w.box))) || bh(e.box);
  const taken = new Set(links.map((l) => l.a));
  const out = [...links];
  const isLabel = (e: Entity) => /\s\/\s/.test(e.text) || /:$/.test(e.text) || (e.text !== e.text.toUpperCase() && e.text.split(' ').length <= 6);
  for (const q of entities) {
    if (!isLabel(q) || q.label === 'other' && /<{2}/.test(q.text)) continue;
    const h = hw(q);
    let best: Entity | null = null, bd = Infinity;
    for (const a of entities) {
      if (a === q || isLabel(a) || /<{2}/.test(a.text)) continue;
      const gap = a.box[1] - q.box[3];
      const aligned = Math.abs(a.box[0] - q.box[0]) < 2.5 * h || hOverlap(a.box, q.box) > 0.3;
      if (gap > -0.3 * h && gap < 1.6 * h && aligned && hw(a) >= 1.05 * h && gap < bd) { bd = gap; best = a; }
    }
    if (!best) continue;
    // replace whatever the classifier linked to this answer
    for (let i = out.length - 1; i >= 0; i--) {
      const l = out[i];
      if (l.a === best.id || l.q === q.id || l.a === q.id || l.q === best.id) out.splice(i, 1);
    }
    q.label = 'question';
    best.label = 'answer';
    out.push({ q: q.id, a: best.id, p: 0.9 });
    taken.add(best.id);
  }
  return out;
}

export async function analyse(words: OcrWord[], W: number, H: number, clf: Classifiers, second?: SecondReading[], kind: 'form' | 'id' = 'form'): Promise<Analysis> {
  const entities = groupWords(words);
  if (!entities.length) return { entities, links: [], fields: [] };
  const ctx = pageCtx(entities, W, H);
  const probs = await clf.entity(entityFeatures(entities, ctx));
  entities.forEach((e, i) => {
    e.probs = probs[i];
    e.label = argmaxLabel(probs[i]);
    // machine-readable zone lines are handled by the MRZ parser, not as form fields
    if (/<{3,}|[A-Z0-9]<{2}[A-Z0-9]/.test(e.text.replace(/\s/g, ''))) {
      e.label = 'other';
      e.probs = [0, 0, 0, 1];
    }
  });
  const idx = (l: string) => entities.map((_, i) => i).filter((i) => entities[i].label === l);
  const pairs = candidatePairs(entities, idx('question'), idx('answer'), ctx);
  const lp = pairs.length ? await clf.link(pairFeatures(entities, pairs, ctx)) : [];
  let links = decodeLinks(pairs, lp);
  if (kind === 'id') {
    links = idLayout(entities, links);
    // anything left labelled "question" without a link on an ID is usually a value or noise
    const linkedQ = new Set(links.map((l) => l.q));
    for (const e of entities) if (e.label === 'question' && !linkedQ.has(e.id) && !/[:/]/.test(e.text)) e.label = 'other';
  }

  let agreement: (boolean | null)[] | undefined;
  const agreeOf = new Map<OcrWord, boolean | null>();
  if (second) {
    agreement = wordAgreement(words, second);
    words.forEach((w, i) => agreeOf.set(w, agreement![i]));
  }

  const qi = LABELS.indexOf('question'), ai = LABELS.indexOf('answer');
  const fields: Field[] = [];
  const answered = new Set<number>();
  const byQ = new Map<number, Link[]>();
  for (const l of links) {
    if (!byQ.has(l.q)) byQ.set(l.q, []);
    byQ.get(l.q)!.push(l);
  }
  for (const [q, ls] of byQ) {
    const Q = entities[q];
    ls.sort((x, y) => entities[x.a].box[1] - entities[y.a].box[1] || entities[x.a].box[0] - entities[y.a].box[0]);
    const A = ls.map((l) => entities[l.a]);
    ls.forEach((l) => answered.add(l.a));
    const aw = A.flatMap((a) => a.words);
    const pLink = Math.min(...ls.map((l) => l.p));
    const pLab = Math.min(Q.probs![qi], ...A.map((a) => a.probs![ai]));
    const ocr = aw.length ? Math.min(...aw.map((w) => w.conf)) / 100 : 0;
    const reasons: string[] = [];
    let conf: number;
    let alt: string | undefined;
    if (second) {
      const ag = aw.map((w) => agreeOf.get(w));
      const known = ag.filter((x) => x !== null && x !== undefined);
      const agree = known.length ? known.filter(Boolean).length / known.length : 0;
      if (ag.some((x) => x === false)) reasons.push('Readers disagree');
      if (known.length < ag.length) reasons.push('Second reader missed text');
      conf = 0.3 * pLink + 0.1 * pLab + 0.25 * ocr + 0.35 * agree;
      const st = A.map((a) => secondText(a.box, second)).join(' ').trim();
      if (st && norm(st) !== norm(A.map((a) => a.text).join(' '))) alt = st;
    } else {
      conf = 0.4 * pLink + 0.15 * pLab + 0.45 * ocr;
    }
    if (ocr < 0.7) reasons.push('Low OCR confidence');
    if (pLink < 0.7) reasons.push('Uncertain question–answer link');
    fields.push({
      key: keyText(Q.text),
      value: A.map((a) => a.text).join(' '),
      qId: Q.id,
      aId: A[0].id,
      confidence: conf,
      status: conf < 0.8 || reasons.length ? 'review' : 'ok',
      reasons,
      alt,
      source: 'layout',
    });
  }
  // questions without an answer: an empty field the clerk may need to fill
  for (const q of idx('question')) {
    if (byQ.has(q)) continue;
    fields.push({ key: keyText(entities[q].text), value: '', qId: entities[q].id, confidence: entities[q].probs![qi], status: 'empty', reasons: ['No answer found'], source: 'layout' });
  }
  fields.sort((x, y) => {
    const a = entities[x.qId!].box, b = entities[y.qId!].box;
    return a[1] - b[1] || a[0] - b[0];
  });
  return { entities, links, fields, agreement };
}
