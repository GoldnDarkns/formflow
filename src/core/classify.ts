// Document-type recognition: decides what a page is (passport, Emirates ID,
// driving licence, visa, form ...) from three kinds of evidence:
//   1. the machine-readable zone, if there is one (strongest signal)
//   2. tell-tale text found by OCR (titles, number formats)
//   3. page layout (shape, amount of text, number of "label:" patterns)
// Each piece of evidence adds weight to a type; a softmax turns the weights into
// a probability, which is the confidence shown to the user. Every decision
// carries its evidence, so the clerk can see why.
import type { OcrWord } from './types';
import type { MrzResult } from './mrz';

export type DocType = 'passport' | 'emirates_id' | 'id_card' | 'driving_licence' | 'visa' | 'form' | 'other';
export const DOC_TYPES: DocType[] = ['passport', 'emirates_id', 'id_card', 'driving_licence', 'visa', 'form', 'other'];
export const DOC_LABEL: Record<DocType, string> = {
  passport: 'Passport',
  emirates_id: 'Emirates ID',
  id_card: 'ID card',
  driving_licence: 'Driving licence',
  visa: 'Visa or permit',
  form: 'Form',
  other: 'Other',
};
export const ID_TYPES: DocType[] = ['passport', 'emirates_id', 'id_card', 'driving_licence', 'visa'];

export interface Evidence {
  type: DocType;
  weight: number;
  why: string;
}
export interface Classification {
  type: DocType;
  confidence: number; // probability of the chosen type
  probs: Record<DocType, number>;
  evidence: Evidence[]; // evidence for the chosen type, strongest first
  all: Evidence[]; // every piece of evidence, for the critic
}

export function linesOf(words: OcrWord[]): string[] {
  const m = new Map<number, OcrWord[]>();
  for (const w of words) {
    if (!m.has(w.line)) m.set(w.line, []);
    m.get(w.line)!.push(w);
  }
  return [...m.values()]
    .map((ws) => ws.sort((a, b) => a.box[0] - b.box[0]))
    .sort((a, b) => a[0].box[1] - b[0].box[1])
    .map((ws) => ws.map((w) => w.text).join(' '));
}

export const EID_RE = /784[\s-]?(\d{4})[\s-]?(\d{7})[\s-]?(\d)/;

export function classifyDocument(words: OcrWord[], W: number, H: number, mrz: MrzResult | null): Classification {
  const text = linesOf(words).join('\n').toUpperCase();
  const ev: Evidence[] = [];
  const add = (type: DocType, weight: number, why: string) => ev.push({ type, weight, why });
  const has = (re: RegExp) => re.test(text);

  // 1. machine-readable zone
  if (mrz) {
    const t = mrz.docType[0];
    const country = mrz.fields.find((f) => f.key === 'Issuing country')?.value;
    const bonus = mrz.valid ? 2 : 0;
    if (mrz.format === 'MRVA' || mrz.format === 'MRVB' || t === 'V') add('visa', 6 + bonus, 'Machine-readable zone in visa format');
    else if (mrz.format === 'TD3' && t === 'P') add('passport', 6 + bonus, `Passport machine-readable zone${mrz.valid ? ', check digits pass' : ''}`);
    else if (mrz.format === 'TD1' && country === 'ARE') add('emirates_id', 6 + bonus, 'ID-card machine-readable zone issued by ARE');
    else if (mrz.format === 'TD1' || mrz.format === 'TD2') add('id_card', 5 + bonus, `ID-card machine-readable zone${country ? ' issued by ' + country : ''}`);
    else add('passport', 3, 'Machine-readable zone found');
  }

  // 2. tell-tale text
  if (has(EID_RE)) add('emirates_id', 5, 'Number in Emirates ID format (784-YYYY-NNNNNNN-N)');
  if (has(/IDENTITY\s+CARD|RESIDENT\s+IDENTITY|EMIRATES\s+ID/)) add('emirates_id', 3, 'Words "identity card"');
  if (has(/UNITED\s+ARAB\s+EMIRATES/) && has(/ID\s+NUMBER|IDENTITY/)) add('emirates_id', 1.5, 'Issued by United Arab Emirates');
  if (has(/IDENTITY\s+CARD|NATIONAL\s+ID/) && !has(/EMIRATES/)) add('id_card', 2.5, 'Words "identity card"');
  if (has(/DRIVING\s+LICEN[CS]E|DRIVER'?S?\s+LICEN[CS]E/)) add('driving_licence', 5, 'Title "driving licence"');
  if (has(/LICEN[CS]E\s+NO|TRAFFIC\s+(CODE|FILE)|PLACE\s+OF\s+ISSUE/)) add('driving_licence', 2, 'Licence number or traffic file');
  if (has(/PERMITTED\s+VEHICLES|VEHICLE\s+(CLASS|CATEGOR)/)) add('driving_licence', 1.5, 'Vehicle categories listed');
  if (has(/\bE-?VISA\b|\bVISA\b|ENTRY\s+PERMIT|RESIDENCE\s+PERMIT|RESIDENCY/)) add('visa', 4, 'Words "visa" or "permit"');
  if (has(/SPONSOR|UID\s+NO|DURATION\s+OF\s+STAY|VALID\s+UNTIL|PERMIT\s+NO|VISA\s+(NO|TYPE|NUMBER)/)) add('visa', 2.5, 'Visa details (sponsor, permit number, validity)');
  if (has(/PASSPORT|PASSEPORT|PASAPORTE/)) {
    // visas and permits also mention the holder's passport number
    const alsoVisa = ev.some((e) => e.type === 'visa');
    add('passport', alsoVisa ? 0.5 : 3, 'Word "passport"');
  }
  if (has(/PLACE\s+OF\s+BIRTH|AUTHORITY|DATE\s+OF\s+ISSUE/) && has(/SURNAME|GIVEN\s+NAMES?/)) add('passport', 1.5, 'Passport data-page labels');

  // 3. layout
  const colons = (text.match(/:(\s|$)/g) ?? []).length;
  const portrait = H > W * 1.15;
  if (colons >= 6) add('form', 2.5, `${colons} "label:" patterns`);
  else if (colons >= 3) add('form', 1, `${colons} "label:" patterns`);
  if (words.length >= 120) add('form', 1.5, `Dense page (${words.length} words)`);
  if (portrait) add('form', 1, 'Portrait page');
  if (has(/\bFORM\b|APPLICATION|FAX|MEMO|REPORT|REQUEST|SIGNATURE/)) add('form', 1.5, 'Form vocabulary');
  if (!portrait && words.length < 90) for (const t of ['passport', 'emirates_id', 'id_card', 'driving_licence'] as DocType[]) add(t, 0.3, 'Card-shaped, little text');
  add('other', 1.2, 'No strong evidence');

  const score = Object.fromEntries(DOC_TYPES.map((t) => [t, 0])) as Record<DocType, number>;
  for (const e of ev) score[e.type] += e.weight;
  const mx = Math.max(...Object.values(score));
  const exps = DOC_TYPES.map((t) => Math.exp(score[t] - mx));
  const Z = exps.reduce((a, b) => a + b, 0);
  const probs = Object.fromEntries(DOC_TYPES.map((t, i) => [t, exps[i] / Z])) as Record<DocType, number>;
  const type = DOC_TYPES.reduce((a, b) => (probs[b] > probs[a] ? b : a));
  return {
    type,
    confidence: probs[type],
    probs,
    evidence: ev.filter((e) => e.type === type && e.weight >= 1).sort((a, b) => b.weight - a.weight),
    all: ev,
  };
}
