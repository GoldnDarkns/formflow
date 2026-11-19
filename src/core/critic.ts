// The critic: a second look at the sorter's decision. It asks three things.
//   1. Does the document contain what this type should contain?
//   2. What argues for a different type (the runner-up)?
//   3. Does an independent vision model, looking only at the picture, agree?
import { DOC_LABEL, EID_RE, linesOf, type Classification, type DocType } from './classify';
import type { Fact } from './facts';
import type { MrzResult } from './mrz';
import type { Field, OcrWord } from './types';
import type { VisionOpinion } from './critic-vision';

export interface Critique {
  verdict: 'holds' | 'doubt' | 'challenged';
  summary: string;
  expected: { ok: boolean; text: string }[];
  against: string[]; // arguments for another type
  runnerUp: { type: DocType; prob: number } | null;
  vision: (VisionOpinion & { agrees: boolean }) | null;
}

export function critique(type: DocType, cls: Classification, facts: Fact[], mrz: MrzResult | null, fields: Field[], words: OcrWord[], vision?: VisionOpinion | null): Critique {
  const has = (k: string) => facts.some((f) => f.key === k && f.value);
  const text = linesOf(words).join(' ');
  const expected: Critique['expected'] = [];
  const want = (ok: boolean, yes: string, no: string) => expected.push({ ok, text: ok ? yes : no });
  if (type === 'passport') {
    want(!!mrz, 'Has a machine-readable zone', 'No machine-readable zone was found, which every passport data page has');
    if (mrz) want(mrz.valid, 'Its check digits pass', 'Its check digits do not all pass');
    want(has('Passport number'), 'Passport number found', 'No passport number found');
    want(has('Expiry date'), 'Expiry date found', 'No expiry date found');
  } else if (type === 'emirates_id') {
    const eid = facts.find((f) => f.key === 'Emirates ID number');
    want(!!eid || mrz?.format === 'TD1', eid ? 'Has a number in Emirates ID format' : 'Has an ID-card machine-readable zone', 'No Emirates ID number (784-…) and no card code lines found');
    if (eid) want(!!eid.valid, 'The ID number checksum passes', 'The ID number checksum fails');
    want(has('Expiry date') || has('Date of birth'), 'Dates found', 'No date of birth or expiry date found');
  } else if (type === 'id_card') {
    want(!!mrz, 'Has ID-card code lines', 'No machine-readable zone found');
    want(has('Document number'), 'Document number found', 'No document number found');
  } else if (type === 'driving_licence') {
    want(has('Licence number'), 'Licence number found', 'No licence number found');
    want(has('Expiry date'), 'Expiry date found', 'No expiry date found');
    want(!mrz, 'No passport-style code lines, as expected', 'Has a machine-readable zone, which licences usually lack');
  } else if (type === 'visa') {
    want(has('Visa number') || !!mrz, 'Visa or permit number found', 'No visa or permit number found');
    want(has('Expiry date'), 'Validity date found', 'No validity date found');
    want(has('Passport number'), "Holder's passport number found", "No passport number found, which visas usually quote");
  } else if (type === 'form') {
    const n = fields.filter((f) => f.value).length;
    want(n >= 3, `${n} label and value pairs found`, `Only ${n} label and value pairs found`);
    want(!mrz && !EID_RE.test(text), 'No identity-document codes on the page', 'Contains identity-document codes, unusual for a form');
  }
  const ranked = (Object.entries(cls.probs) as [DocType, number][]).filter(([t]) => t !== type).sort((a, b) => b[1] - a[1]);
  const runnerUp = ranked.length && ranked[0][1] > 0.005 ? { type: ranked[0][0], prob: ranked[0][1] } : null;
  const against = runnerUp ? cls.all.filter((e) => e.type === runnerUp.type && e.weight >= 1).sort((a, b) => b.weight - a.weight).map((e) => `${e.why} points to ${DOC_LABEL[runnerUp.type].toLowerCase()}`) : [];
  const v = vision ? { ...vision, agrees: vision.type === type || (type === 'id_card' && vision.type === 'emirates_id') } : null;

  const failed = expected.filter((e) => !e.ok).length;
  let verdict: Critique['verdict'] = 'holds';
  // the picture-only critic is the weaker witness: it raises doubt only when the
  // sorter is not already backed by strong evidence and passed expectations
  const visionDoubt = !!v && !v.agrees && (failed > 0 || cls.confidence < 0.95);
  if (failed > 0 || visionDoubt || cls.confidence < 0.8) verdict = 'doubt';
  if ((expected.length && failed > expected.length / 2) || (v && !v.agrees && v.confidence >= 0.6 && (failed > 0 || cls.confidence < 0.9))) verdict = 'challenged';
  // "an Emirates ID", "an ID card", "a passport"
  const label = DOC_LABEL[type].replace(/^[A-Z][a-z]/, (m) => m.toLowerCase());
  const name = label.replace(/^emirates/, 'Emirates');
  const an = /^(Emirates|ID)/.test(name) ? 'an' : 'a';
  const summary =
    verdict === 'holds' ? (v && !v.agrees ? `Everything ${an} ${name} should contain is here. The picture-only model guessed differently, but the checks outweigh it.` : `The critic agrees this is ${an} ${name}.`)
    : verdict === 'doubt' ? `Probably ${an} ${name}, with ${failed ? 'something missing' : visionDoubt ? 'a differing second opinion' : 'a narrow margin'}.`
    : `The critic doubts this is ${an} ${name}. Check the type.`;
  return { verdict, summary, expected, against, runnerUp, vision: v };
}
