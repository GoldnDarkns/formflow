// Type-specific "key facts": the handful of values a clerk actually needs from
// an identity document, normalised (ISO dates) and verified where a checksum
// exists (MRZ check digits, Emirates ID Luhn digit).
import type { Field, OcrWord } from './types';
import type { MrzResult } from './mrz';
import { EID_RE, linesOf, type DocType } from './classify';

export interface Fact {
  key: string;
  value: string;
  proof?: 'check digit' | 'checksum'; // verified by arithmetic, not by a model
  valid?: boolean;
  from: 'mrz' | 'text';
}

const MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
const iso = (y: number, m: number, d: number) => (m >= 1 && m <= 12 && d >= 1 && d <= 31 && y > 1900 && y < 2100 ? `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}` : null);

/** First date found in a string, as YYYY-MM-DD. Day-first for numeric dates. */
export function parseDate(s: string): string | null {
  let m = s.match(/\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b/);
  if (m) return iso(+m[1], +m[2], +m[3]);
  m = s.match(/\b(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{4})\b/);
  if (m) return iso(+m[3], +m[2], +m[1]);
  m = s.match(/\b(\d{1,2})[\s-]*([A-Za-z]{3})[A-Za-z]*[\s-]*(\d{4})\b/);
  if (m) {
    const mo = MONTHS.indexOf(m[2].toUpperCase());
    if (mo >= 0) return iso(+m[3], mo + 1, +m[1]);
  }
  return null;
}

export function luhnOk(digits: string): boolean {
  let sum = 0;
  for (let i = 0; i < digits.length; i++) {
    let d = +digits[digits.length - 1 - i];
    if (i % 2 === 1) { d *= 2; if (d > 9) d -= 9; }
    sum += d;
  }
  return digits.length > 1 && sum % 10 === 0;
}

function labelled(lines: string[], fields: Field[], label: RegExp, pick: (s: string) => string | null): string | null {
  for (const f of fields) if (label.test(f.key) && f.value) { const v = pick(f.value); if (v) return v; }
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(label);
    if (!m) continue;
    const rest = lines[i].slice(m.index! + m[0].length);
    for (const cand of [rest, lines[i + 1] ?? '', lines[i + 2] ?? '']) { const v = pick(cand); if (v) return v; }
  }
  return null;
}
const textValue = (s: string) => {
  const v = s.replace(/^[\s:./-]+/, '').replace(/\s{2,}.*$/, '').trim();
  return v.length >= 2 && !/^(no|number|date)\b/i.test(v) ? v : null;
};

/** Value after a numbered field tag at the start of a line, as on EU-style licences ("4b. 02-07-2027"). */
function numbered<T>(lines: string[], tag: string, pick: (s: string) => T | null): T | null {
  const re = new RegExp('^\\s*' + tag.replace('.', '') + '[.,:]?\\s+(.+)$', 'i');
  for (const l of lines) { const m = l.match(re); if (m) { const v = pick(m[1]); if (v) return v; } }
  return null;
}
/** a person's name as printed: letters, spaces, hyphens; stops at the next label or a number */
const nameValue = (s: string) => {
  const v = s.replace(/^[\s:./-]+/, '').match(/^[\p{L}][\p{L} '\-]{1,48}/u)?.[0].trim() ?? '';
  return v.length >= 2 && !/^(no|number|date|nom|name|surname|given|pr[eé]noms?)$/i.test(v) ? v : null;
};
/** a document number: letters and digits with at least one digit, so a word such as "IDENTITY" is never taken for one */
const codeValue = (min: number) => (s: string) => (s.replace(/^[\s:./-]+/, '').match(new RegExp('\\b[A-Z0-9][A-Z0-9-]{' + (min - 1) + ',}\\b', 'g')) ?? []).find((v) => /\d/.test(v)) ?? null;

export function extractFacts(type: DocType, words: OcrWord[], fields: Field[], mrz: MrzResult | null): Fact[] {
  const lines = linesOf(words);
  const facts: Fact[] = [];
  const have = (k: string) => facts.some((f) => f.key === k);
  const add = (f: Fact) => { if (f.value && !have(f.key)) facts.push(f); };

  if (mrz) {
    const g = (k: string) => mrz.fields.find((f) => f.key === k);
    const name = [g('Given names')?.value, g('Surname')?.value].filter(Boolean).join(' ');
    add({ key: 'Full name', value: name, from: 'mrz' });
    const num = g('Document number');
    if (num) add({ key: type === 'passport' ? 'Passport number' : 'Document number', value: num.value, proof: 'check digit', valid: num.valid, from: 'mrz' });
    add({ key: 'Nationality', value: g('Nationality')?.value ?? '', from: 'mrz' });
    const dob = g('Date of birth'), exp = g('Expiry date');
    if (dob) add({ key: 'Date of birth', value: dob.value, proof: 'check digit', valid: dob.valid, from: 'mrz' });
    add({ key: 'Sex', value: g('Sex')?.value ?? '', from: 'mrz' });
    if (exp) add({ key: 'Expiry date', value: exp.value, proof: 'check digit', valid: exp.valid, from: 'mrz' });
  }

  if (type === 'emirates_id') {
    const all = lines.join(' ') + ' ' + (mrz?.lines.join(' ').replace(/</g, ' ') ?? '');
    const m = all.match(EID_RE);
    if (m) {
      const digits = '784' + m[1] + m[2] + m[3];
      add({ key: 'Emirates ID number', value: `784-${m[1]}-${m[2]}-${m[3]}`, proof: 'checksum', valid: luhnOk(digits), from: 'text' });
    }
  }
  if (type === 'driving_licence') {
    const no = labelled(lines, fields, /number\s+of\s+national\s+licen[cs]e|licen[cs]e\s*no\.?|licen[cs]e\s*number/i, codeValue(5)) ?? numbered(lines, '5', codeValue(5));
    if (no) add({ key: 'Licence number', value: no, from: 'text' });
    // EU-style cards number their fields instead of labelling them: 1 surname, 2 given names, 3 birth, 4a issue, 4b expiry
    const sur = numbered(lines, '1', nameValue), giv = numbered(lines, '2', nameValue);
    if (sur && giv) add({ key: 'Full name', value: `${giv} ${sur}`, from: 'text' });
    const dob = numbered(lines, '3', parseDate), iss = numbered(lines, '4a', parseDate), exp = numbered(lines, '4b', parseDate);
    if (dob) add({ key: 'Date of birth', value: dob, from: 'text' });
    if (iss) add({ key: 'Issue date', value: iss, from: 'text' });
    if (exp) add({ key: 'Expiry date', value: exp, from: 'text' });
    const place = labelled(lines, fields, /place\s+of\s+issue/i, textValue);
    if (place) add({ key: 'Place of issue', value: place, from: 'text' });
  }
  if (type === 'visa') {
    const no = labelled(lines, fields, /(visa|permit|entry permit)\s*(no\.?|number)/i, (s) => s.match(/\b[A-Z0-9][A-Z0-9/-]{5,}\b/)?.[0] ?? null);
    if (no) add({ key: 'Visa number', value: no, from: 'text' });
    const pp = labelled(lines, fields, /passport\s*(no\.?|number)/i, (s) => s.match(/\b[A-Z0-9]{6,10}\b/)?.[0] ?? null);
    if (pp) add({ key: 'Passport number', value: pp, from: 'text' });
  }
  if (type === 'vehicle_licence') {
    const plate = labelled(lines, fields, /traffic\s+plate\s+no\.?|plate\s+(no\.?|number)/i, (s) => s.match(/\b[A-Z]{1,3}\s*\/?\s*\d{3,6}\b/)?.[0].replace(/\s+/g, ' ') ?? null);
    if (plate) add({ key: 'Plate number', value: plate, from: 'text' });
    const owner = labelled(lines, fields, /\bowner\b/i, textValue);
    if (owner) add({ key: 'Owner', value: owner, from: 'text' });
    const chassis = labelled(lines, fields, /chassis\s+no\.?/i, (s) => s.match(/\b[A-HJ-NPR-Z0-9]{11,17}\b/)?.[0] ?? null);
    if (chassis) add({ key: 'Chassis number', value: chassis, from: 'text' });
    const engine = labelled(lines, fields, /eng(ine)?\.?\s+no\.?/i, codeValue(6));
    if (engine) add({ key: 'Engine number', value: engine, from: 'text' });
    const veh = labelled(lines, fields, /veh(icle)?\.?\s+type/i, (s) => s.replace(/^[\s:./-]+/, '').match(/^[A-Z][A-Z0-9]+(\s[A-Z0-9]+)?/)?.[0] ?? null);
    if (veh) add({ key: 'Vehicle', value: veh, from: 'text' });
    const year = labelled(lines, fields, /\bmodel\b/i, (s) => s.match(/\b(19|20)\d{2}\b/)?.[0] ?? null);
    if (year) add({ key: 'Model year', value: year, from: 'text' });
    const reg = labelled(lines, fields, /reg\.?\s+date/i, parseDate);
    if (reg) add({ key: 'Registration date', value: reg, from: 'text' });
    const ins = labelled(lines, fields, /ins\.?\s+exp\.?/i, parseDate);
    if (ins) add({ key: 'Insurance expiry', value: ins, from: 'text' });
    const exp = labelled(lines, fields, /(?<!ins\.?\s)exp\.?\s+date/i, parseDate);
    if (exp) add({ key: 'Expiry date', value: exp, from: 'text' });
  }
  if (type === 'id_card') {
    const no = labelled(lines, fields, /id\s+number|national\s+id\b|document\s+no\.?|card\s+no\.?|\bno\.(?=\s|$)/i, codeValue(6));
    if (no) add({ key: 'Document number', value: no, from: 'text' });
  }
  if (type !== 'form' && type !== 'other') {
    // surname and given names printed separately, in whatever language the label is
    const sur = labelled(lines, fields, /surname|family\s+name|apellidos?|sobrenome|\bnom\b/i, nameValue);
    const giv = labelled(lines, fields, /given\s+names?|first\s+names?|pr[eé]noms?|nombres?/i, nameValue);
    if (sur && giv) add({ key: 'Full name', value: `${giv} ${sur}`, from: 'text' });
    const name = labelled(lines, fields, /^\s*(full\s+)?name\b|\bname\s*:/i, textValue);
    if (name) add({ key: 'Full name', value: name, from: 'text' });
    const nat = labelled(lines, fields, /nationality/i, (s) => s.match(/[A-Za-z][A-Za-z ]{2,}/)?.[0].trim() ?? null);
    if (nat) add({ key: 'Nationality', value: nat, from: 'text' });
    const dob = labelled(lines, fields, /date\s+of\s+birth|\bDOB\b|birth\s+date/i, parseDate);
    if (dob) add({ key: 'Date of birth', value: dob, from: 'text' });
    const iss = labelled(lines, fields, /date\s+of\s+issue|issue\s+date|issuing\s+date/i, parseDate);
    if (iss) add({ key: 'Issue date', value: iss, from: 'text' });
    const exp = labelled(lines, fields, /date\s+of\s+expiry|expiry(\s+date)?|expiration|valid\s+until|\buntil\b|expires/i, parseDate);
    if (exp) add({ key: 'Expiry date', value: exp, from: 'text' });
  }
  if (type !== 'form' && type !== 'other' && type !== 'vehicle_licence') {
    // Dates nobody labelled: on an identity document the earliest date is the birth date and the
    // latest is the expiry, with the issue date between them. Used only for dates still missing.
    const dates = [...new Set(lines.map(parseDate).filter((d): d is string => !!d))].sort();
    if (dates.length >= 2) {
      const first = dates[0], last = dates[dates.length - 1];
      const years = +last.slice(0, 4) - +first.slice(0, 4);
      if (years >= 10) {
        add({ key: 'Date of birth', value: first, from: 'text' });
        add({ key: 'Expiry date', value: last, from: 'text' });
        if (dates.length >= 3) add({ key: 'Issue date', value: dates[dates.length - 2], from: 'text' });
      }
    }
  }
  if (type === 'driving_licence') {
    // EU-style card whose field numbers were not read: the two name rows sit directly above the birth-date row
    const i = lines.findIndex((l) => parseDate(l) && parseDate(l) === facts.find((f) => f.key === 'Date of birth')?.value);
    const plain = (l?: string) => (l && /^[\p{L}][\p{L} '\-]{1,40}$/u.test(l.replace(/^\d[ab]?[.,]?\s+/, '').trim()) ? l.replace(/^\d[ab]?[.,]?\s+/, '').trim() : null);
    if (i >= 2 && plain(lines[i - 1]) && plain(lines[i - 2]) && !/licen[cs]e|permis|republic|kingdom|states|emirates/i.test(lines[i - 2])) add({ key: 'Full name', value: `${plain(lines[i - 1])} ${plain(lines[i - 2])}`, from: 'text' });
    // a licence number standing on a row of its own
    const solo = lines.map((l) => l.replace(/^\d[ab]?[.,]?\s+/, '').trim()).find((l) => /^[A-Z0-9]{6,12}$/.test(l) && /\d/.test(l) && !parseDate(l));
    if (solo) add({ key: 'Licence number', value: solo, from: 'text' });
  }
  return facts;
}

export interface Alert {
  level: 'bad' | 'warn' | 'good';
  text: string;
}

export function alertsFor(type: DocType, typeConfidence: number, facts: Fact[], mrz: MrzResult | null, today = new Date()): Alert[] {
  const out: Alert[] = [];
  const exp = facts.find((f) => f.key === 'Expiry date')?.value;
  if (exp && /^\d{4}-\d{2}-\d{2}$/.test(exp)) {
    const days = Math.round((new Date(exp + 'T00:00:00').getTime() - today.getTime()) / 864e5);
    if (days < 0) out.push({ level: 'bad', text: `Expired on ${exp}` });
    else if (days < 183) out.push({ level: 'warn', text: `Expires in ${days} days${type === 'passport' ? '. Many countries ask for six months of validity' : ''}` });
    else out.push({ level: 'good', text: `Valid until ${exp}` });
  }
  if (mrz && !mrz.valid) out.push({ level: 'bad', text: `Check digit failed: ${mrz.checks.filter((c) => !c.ok).map((c) => c.name.toLowerCase()).join(', ')}` });
  if (mrz?.valid) out.push({ level: 'good', text: 'All check digits pass' });
  const eid = facts.find((f) => f.key === 'Emirates ID number');
  if (eid) out.push(eid.valid ? { level: 'good', text: 'Emirates ID checksum passes' } : { level: 'warn', text: 'Emirates ID checksum fails. A digit is probably misread' });
  if (typeConfidence < 0.7) out.push({ level: 'warn', text: 'Not sure about the document type. Check it' });
  return out;
}
