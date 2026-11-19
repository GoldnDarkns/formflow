// Document-type recognition: decides what a page is (passport, Emirates ID,
// driving licence, vehicle licence, visa, form ...).
//
// A page is described by a fixed list of clues (features): is there a
// machine-readable zone and of what kind, do certain tell-tale words or number
// formats appear, what shape is the page. Every clue has a weight for every
// document type; the weights of the clues that fire are added up and a softmax
// turns the totals into a probability, which is the confidence shown.
//
// Two sets of weights exist:
//   PRIOR   hand-set, from reading the documents (the original rule-based sorter)
//   learned fitted on real documents by scripts/train-sorter.ts, pulled towards
//           PRIOR so that types with no real examples keep their hand-set rules
//
// The clues are generic words and formats chosen by hand. Nothing read from a
// training document (no name, number or address) can end up in the model: only
// one number per clue and type is stored.
import type { OcrWord } from './types';
import type { MrzResult } from './mrz';
import learned from './sorter.json' with { type: 'json' };

export type DocType = 'passport' | 'emirates_id' | 'id_card' | 'driving_licence' | 'vehicle_licence' | 'visa' | 'form' | 'other';
export const DOC_TYPES: DocType[] = ['passport', 'emirates_id', 'id_card', 'driving_licence', 'vehicle_licence', 'visa', 'form', 'other'];
export const DOC_LABEL: Record<DocType, string> = {
  passport: 'Passport',
  emirates_id: 'Emirates ID',
  id_card: 'ID card',
  driving_licence: 'Driving licence',
  vehicle_licence: 'Vehicle licence',
  visa: 'Visa or permit',
  form: 'Form',
  other: 'Other',
};
export const ID_TYPES: DocType[] = ['passport', 'emirates_id', 'id_card', 'driving_licence', 'vehicle_licence', 'visa'];

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

interface Ctx { text: string; words: OcrWord[]; W: number; H: number; mrz: MrzResult | null; mrzOk: boolean; mrzCountry?: string }
interface Clue { id: string; why: string; test: (c: Ctx) => boolean; prior: Partial<Record<DocType, number>> }
const has = (re: RegExp) => (c: Ctx) => re.test(c.text);

/** The clues, with their hand-set weights. Order is the feature order of the learned model. */
export const CLUES: Clue[] = [
  // 1. machine-readable zone (counted only when at least one of its check digits adds up, or it is fully valid)
  { id: 'mrz_passport', why: 'Passport machine-readable zone', test: (c) => c.mrzOk && c.mrz!.format === 'TD3' && c.mrz!.docType[0] === 'P', prior: { passport: 6 } },
  { id: 'mrz_visa', why: 'Machine-readable zone in visa format', test: (c) => c.mrzOk && (c.mrz!.format === 'MRVA' || c.mrz!.format === 'MRVB' || c.mrz!.docType[0] === 'V'), prior: { visa: 6 } },
  { id: 'mrz_card_are', why: 'ID-card machine-readable zone issued by ARE', test: (c) => c.mrzOk && c.mrz!.format === 'TD1' && c.mrzCountry === 'ARE', prior: { emirates_id: 6 } },
  { id: 'mrz_card', why: 'ID-card machine-readable zone', test: (c) => c.mrzOk && (c.mrz!.format === 'TD1' || c.mrz!.format === 'TD2') && c.mrzCountry !== 'ARE' && c.mrz!.docType[0] !== 'V', prior: { id_card: 5 } },
  { id: 'mrz_valid', why: 'Every check digit of the machine-readable zone passes', test: (c) => !!c.mrz?.valid, prior: { passport: 1, emirates_id: 1, id_card: 1, visa: 1 } },
  { id: 'chevrons', why: 'Rows of "<" filler characters', test: (c) => (c.text.match(/</g)?.length ?? 0) >= 6, prior: { passport: 1, emirates_id: 0.5, id_card: 0.5 } },
  // 2. tell-tale text
  { id: 'eid_number', why: 'Number in Emirates ID format (784-YYYY-NNNNNNN-N)', test: has(EID_RE), prior: { emirates_id: 5 } },
  { id: 'eid_title', why: 'Words "identity card"', test: has(/IDENTITY\s+CARD|RESIDENT\s+IDENTITY|EMIRATES\s+ID|GOLDEN\s+CARD/), prior: { emirates_id: 3, id_card: 1 } },
  { id: 'eid_back', why: 'Card number, occupation and issuing place (back of an Emirates ID)', test: has(/CARD\s+NUMBER|OCCUPATION|ISSUING\s+PLACE|EMPLOYER/), prior: { emirates_id: 2 } },
  { id: 'uae', why: 'Issued by United Arab Emirates', test: has(/UNITED\s+ARAB\s+EMIRATES|\bUAE\b/), prior: { emirates_id: 0.6, driving_licence: 0.3, vehicle_licence: 0.3 } },
  { id: 'id_number', why: 'Words "ID number"', test: has(/ID\s+NUMBER|NATIONAL\s+ID/), prior: { emirates_id: 1, id_card: 1.5 } },
  { id: 'dl_title', why: 'Title "driving licence" or "driving permit"', test: has(/DRIVING\s+LICEN[CS]E|DRIVER'?S?\s+LICEN[CS]E|DRIVING\s+PERMIT|PERMIS\s+(INTERNATIONAL\s+)?DE\s+CONDUIRE|VOZA[CČ]KA\s+DOZVOLA|HABILITA[CÇ][AÃ]O|PERMISO\s+DE\s+CONDUC|F[UÜ]HRERSCHEIN|ВОЗАЧКА/), prior: { driving_licence: 5 } },
  { id: 'dl_detail', why: 'Licence number, traffic code or place of issue', test: has(/LICEN[CS]E\s+NO|TRAFFIC\s+(CODE|FILE)|PLACE\s+OF\s+ISSUE|LICENSING\s+AUTHORITY/), prior: { driving_licence: 2 } },
  { id: 'dl_vehicles', why: 'Vehicle categories listed', test: has(/PERMITTED\s+VEHICLES|LIGHT\s+VEHICLE|MOTOR\s?CYCLE|VEHICLE\s+(CLASS|CATEGOR)|CAT[EÉ]GORIES?\s+DE\s+V[EÉ]HICULES|KATEGORIJ/), prior: { driving_licence: 2.5 } },
  { id: 'dl_road', why: 'Road-traffic wording', test: has(/ROAD\s+TRAFFIC|MOTOR\s+TRAFFIC|CIRCULATION\s+ROUTI[EÈ]RE|AMBULANCE|CIVIL\s+DEFEN[CS]E/), prior: { driving_licence: 2 } },
  { id: 'vl_title', why: 'Title "vehicle licence" or "vehicle information"', test: has(/VEHICLE\s+LICEN[CS]E|VEHICLE\s+INFORMATION|REGISTRATION\s+CARD/), prior: { vehicle_licence: 5, driving_licence: -2 } },
  { id: 'vl_detail', why: 'Plate, chassis, engine or insurance details', test: has(/TRAFFIC\s+PLATE|CHASSIS|ENG\.?\s*NO|ENGINE\s+NO|INS\.?\s*EXP|MORTGAGE|G\.?\s?V\.?\s?W|EMPTY\s+WEIGHT|NUM\.?\s+OF\s+PASS|POLICY\s+NO|VEH\.?\s+TYPE/), prior: { vehicle_licence: 4, driving_licence: -1.5 } },
  { id: 'visa_word', why: 'Words "visa" or "permit"', test: has(/\bE-?VISA\b|\bVISA\b|ENTRY\s+PERMIT|RESIDENCE\s+PERMIT|RESIDENCY/), prior: { visa: 4, passport: -2.5 } },
  { id: 'visa_detail', why: 'Visa details (sponsor, permit number, validity)', test: has(/SPONSOR|UID\s+NO|DURATION\s+OF\s+STAY|VALID\s+UNTIL|PERMIT\s+NO|VISA\s+(NO|TYPE|NUMBER)/), prior: { visa: 2.5 } },
  { id: 'passport_word', why: 'Word "passport"', test: has(/PASSPORT|PASSEPORT|PASAPORTE|PASSAPORTE|PA[SŞ]APORT|REISEPASS|ПАСПОРТ|ПАСОШ/), prior: { passport: 3 } },
  { id: 'passport_labels', why: 'Passport data-page labels', test: (c) => /PLACE\s+OF\s+BIRTH|AUTHORITY|DATE\s+OF\s+ISSUE|DATE\s+OF\s+EXPIRY/.test(c.text) && /SURNAME|GIVEN\s+NAMES?|NATIONALITY/.test(c.text), prior: { passport: 1.5 } },
  { id: 'agreement', why: 'Agreement or contract wording', test: has(/AGREEMENT|TERMS\s+AND\s+CONDITIONS|HEREBY|HIRER|CONTRACT/), prior: { form: 2.5 } },
  { id: 'form_words', why: 'Form vocabulary', test: has(/\bFORM\b|APPLICATION|FAX|MEMO|REPORT|REQUEST|SIGNATURE/), prior: { form: 1.5 } },
  // 3. layout
  { id: 'colons_many', why: 'Six or more "label:" patterns', test: (c) => (c.text.match(/:(\s|$)/g) ?? []).length >= 6, prior: { form: 2.5 } },
  { id: 'colons_some', why: 'Three to five "label:" patterns', test: (c) => { const n = (c.text.match(/:(\s|$)/g) ?? []).length; return n >= 3 && n < 6; }, prior: { form: 1 } },
  { id: 'dense', why: 'Dense page (120 words or more)', test: (c) => c.words.length >= 120, prior: { form: 1.5 } },
  { id: 'very_dense', why: 'Very dense page (300 words or more)', test: (c) => c.words.length >= 300, prior: { form: 1 } },
  { id: 'portrait', why: 'Portrait page', test: (c) => c.H > c.W * 1.15, prior: { form: 1 } },
  { id: 'card', why: 'Card-shaped, little text', test: (c) => c.H <= c.W * 1.15 && c.words.length < 90, prior: { passport: 0.3, emirates_id: 0.3, id_card: 0.3, driving_licence: 0.3, vehicle_licence: 0.3 } },
  { id: 'card_shape', why: 'Wider than tall', test: (c) => c.W > c.H * 1.25, prior: {} },
  { id: 'sparse', why: 'Hardly any text', test: (c) => c.words.length < 25, prior: { other: 0.8 } },
];
const BIAS_PRIOR: Partial<Record<DocType, number>> = { other: 1.2 };

export interface SorterModel { fitted: boolean; note?: string; clues: string[]; types: DocType[]; weights: number[][]; bias: number[] }
/** The hand-set weights as a model (rows = types, columns = clues). */
export function priorModel(): SorterModel {
  return { fitted: false, clues: CLUES.map((c) => c.id), types: DOC_TYPES, weights: DOC_TYPES.map((t) => CLUES.map((c) => c.prior[t] ?? 0)), bias: DOC_TYPES.map((t) => BIAS_PRIOR[t] ?? 0) };
}
const LEARNED = learned as SorterModel;
/** The learned weights when they match the current clue list, else the hand-set ones. */
export function activeModel(): SorterModel {
  const ok = LEARNED.fitted && LEARNED.clues.length === CLUES.length && LEARNED.clues.every((id, i) => id === CLUES[i].id) && LEARNED.types.join() === DOC_TYPES.join();
  return ok ? LEARNED : priorModel();
}

/** Which clues fire on this page (1 or 0, in CLUES order). */
export function sorterFeatures(words: OcrWord[], W: number, H: number, mrz: MrzResult | null): number[] {
  const mrzOk = !!mrz && (mrz.valid || mrz.checks.some((c) => c.ok));
  const ctx: Ctx = { text: linesOf(words).join('\n').toUpperCase(), words, W, H, mrz, mrzOk, mrzCountry: mrz?.fields.find((f) => f.key === 'Issuing country')?.value };
  return CLUES.map((c) => (c.test(ctx) ? 1 : 0));
}

export function classifyFeatures(x: number[], model: SorterModel = activeModel()): Classification {
  const score = model.types.map((_, k) => model.bias[k] + x.reduce((a, v, j) => a + v * model.weights[k][j], 0));
  const mx = Math.max(...score);
  const exps = score.map((s) => Math.exp(s - mx));
  const Z = exps.reduce((a, b) => a + b, 0);
  const probs = Object.fromEntries(model.types.map((t, i) => [t, exps[i] / Z])) as Record<DocType, number>;
  const type = model.types.reduce((a, b) => (probs[b] > probs[a] ? b : a));
  const all: Evidence[] = [];
  model.types.forEach((t, k) => x.forEach((v, j) => { const w = v * model.weights[k][j]; if (w >= 0.25) all.push({ type: t, weight: w, why: CLUES[j].why }); }));
  all.push({ type: 'other', weight: Math.max(0.3, model.bias[model.types.indexOf('other')]), why: 'No strong evidence' });
  return { type, confidence: probs[type], probs, evidence: all.filter((e) => e.type === type && e.weight >= 1).sort((a, b) => b.weight - a.weight), all };
}

export function classifyDocument(words: OcrWord[], W: number, H: number, mrz: MrzResult | null, model: SorterModel = activeModel()): Classification {
  return classifyFeatures(sorterFeatures(words, W, H, mrz), model);
}
