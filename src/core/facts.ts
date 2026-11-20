// Type-specific "key facts": the handful of values a clerk actually needs from
// an identity document, normalised (ISO dates) and verified where a checksum
// exists (MRZ check digits, Emirates ID Luhn digit).
import type { Field, OcrWord } from './types';
import type { MrzResult } from './mrz';
import { EID_RE, linesOf, rowsOf, type DocType } from './classify';

export interface Fact {
  key: string;
  value: string;
  proof?: 'check digit' | 'checksum'; // verified by arithmetic, not by a model
  valid?: boolean;
  from: 'mrz' | 'text';
  /** the same value was found in two independent places on the page (code line and printed text) */
  twice?: boolean;
  /** what the other reader saw, where the two readings differ */
  alt?: string;
}

/** Edit distance between two short strings (insert, delete, substitute). */
export function editDistance(a: string, b: string): number {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return prev[b.length];
}
/** words printed on a document that are never part of a holder's name */
const NOT_A_NAME = /^(DATE|DATA|NAMES?|GIVEN|SURNAME|BIRTH|PLACE|SEX|EXPIRY|ISSUE[D]?|NATIONALITY|NUMBER|CARD|HOLDER|SIGNATURE|AUTHORITY|VALID|UNTIL|PASSPORT|LICEN[CS]E|IDENTITY|SPECIMEN|REPUBLIC|KINGDOM|TYPE|CODE|COUNTRY)$/;
const lettersOnly = (s: string) => s.toLowerCase().replace(/[^a-z]/g, '');

/**
 * A name read from the code line has no check digit, and the reader sometimes
 * drops or swaps a letter there. The same name is nearly always printed on the
 * page as well, so each part is compared with the words on the page: an exact
 * match confirms it, and where there is none but exactly one page word is very
 * close and at least as long, that word is used instead.
 */
export function mendName(name: string, words: OcrWord[]): { name: string; twice: boolean } {
  // printed: every word on the page outside the code lines (an exact match confirms a name part);
  // sure: the printed ones read clearly enough to correct with; code: name parts as any reader saw them in a code line
  const printed = new Set<string>(), sure = new Set<string>(), code = new Set<string>();
  for (const w of words) {
    const up = w.text.toUpperCase();
    if (!up.includes('<')) { for (const t of up.split(/[^A-Z]+/)) if (t.length >= 2) { printed.add(t); if (w.conf >= 75) sure.add(t); } continue; }
    // a code line read by the first pass: its padding is often misread as K and L stuck to the name
    for (const t of up.split('<')) { const v = t.replace(/[^A-Z]/g, '').replace(/^[KL]+(?=[A-Z]{3})/, ''); if (v.length >= 2) { code.add(v); if (v.length >= 6) code.add(v.slice(3)); } }
  }
  const isSub = (short: string, long: string) => { let k = 0; for (const ch of long) if (ch === short[k]) k++; return k === short.length; };
  let confirmed = 0;
  const parts = name.split(' ').filter(Boolean);
  const out = parts.map((tok) => {
    if (printed.has(tok)) { confirmed++; return tok; }
    if (tok.length < 2) return tok;
    // padding or the country code stuck to the front of the name: the printed word is the end of this one
    const tails = [...sure].filter((c) => c.length >= 3 && tok.length - c.length >= 1 && tok.length - c.length <= 3 && tok.endsWith(c));
    if (tails.length === 1) return tails[0];
    // padding misread as K or L stuck to the end
    const heads = [...sure].filter((c) => c.length >= 3 && tok.startsWith(c) && /^[KL]{1,3}$/.test(tok.slice(c.length)));
    if (heads.length === 1) return heads[0];
    let best: string[] = [], bd = Infinity;
    for (const c of sure) {
      if (c.length < Math.max(3, tok.length) || c.length > tok.length * 2 + 1 || NOT_A_NAME.test(c)) continue;
      // letters dropped by the reader: everything it did read appears, in order, in the printed word
      const sub = isSub(tok, c);
      const d = sub ? c.length - tok.length : editDistance(tok, c);
      // a part a reader saw in the code line exactly like this is only ever missing letters, never holding wrong ones
      if (sub ? d > (tok.length < 3 ? 0 : tok.length < 4 ? 2 : 3) : code.has(tok) || tok.length < 4 || c.length > tok.length + 3 || d > Math.max(1, Math.round(c.length * 0.4))) continue;
      if (d < bd) { bd = d; best = [c]; } else if (d === bd) best.push(c);
    }
    if (best.length === 1) return best[0];
    if (code.has(tok)) return tok;
    // the other reader saw a longer form of this part in the code line: a dropped letter, not padding
    const extra = (c: string) => { let k = 0, x = ''; for (const ch of c) { if (ch === tok[k]) k++; else x += ch; } return x; };
    const longer = [...code].filter((c) => c.length > tok.length && c.length - tok.length <= 2 && isSub(tok, c) && !/^[KLC]+$/.test(extra(c)));
    return longer.length === 1 ? longer[0] : tok;
  });
  return { name: out.join(' '), twice: parts.length > 0 && confirmed === parts.length };
}

/** Does the start of a row read like one of these labels, allowing for a few misread letters? Returns the rest of the row. */
function afterFuzzyLabel(line: string, labels: string[]): string | null {
  const toks = line.replace(/^\s*\d[ab]?[.,:]?\s+/, '').split(/\s+/).filter(Boolean);
  for (let k = Math.min(3, toks.length); k >= 1; k--) {
    const head = lettersOnly(toks.slice(0, k).join(''));
    if (head.length < 3) continue;
    for (const l of labels) {
      const t = lettersOnly(l);
      if (Math.abs(head.length - t.length) <= 2 && editDistance(head, t) <= Math.floor(t.length * 0.34)) return toks.slice(k).join(' ');
    }
  }
  return null;
}
function fuzzyLabelled<T>(lines: string[], labels: string[], pick: (s: string) => T | null): T | null {
  for (let i = 0; i < lines.length; i++) {
    const rest = afterFuzzyLabel(lines[i], labels);
    if (rest === null) continue;
    for (const cand of rest ? [rest] : [lines[i + 1] ?? '']) { const v = pick(cand); if (v) return v; }
  }
  return null;
}
const HEADER_WORDS = /REPUBLI|BUNDES|DEUTSCH|REINO|ESPANA|FRANCAISE|ITALIANA|KINGDOM|STATES|UNITED|EMIRATES|IDENTITY|NATIONAL|CARD|LICEN[CS]E|PERMI[TS]|PASSPORT|PASSEPORT|SPECIMEN|FEDERA|ISLAMIC|DRIVING|AUTHORITY|REGISTRY|MINISTRY|RESIDEN|TRAFFIC|VISA|ENTRY|TOURIST|FICTIONAL/;

/** the line every international driving permit carries: "Convention on Road Traffic of 8 November 1968" */
const CONVENTION = /convention|road\s+traffic|(?<!\d)19(49|68)(?!\d)/i;
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
  // a label in ordinary case followed by a value in capitals ("Surname BIANCHI"): the capitals are the value
  const caps = s.match(/[a-z][^\s]*\s+(?:[^\p{L}\s]+\s+)*(\p{Lu}[\p{Lu}'\-]+(?: \p{Lu}[\p{Lu}'\-]+)*)\s*$/u)?.[1];
  const v = caps ?? s.replace(/^[^\p{L}]+/u, '').match(/^[\p{L}][\p{L} '\-]{1,48}/u)?.[0].trim() ?? '';
  return v.length >= 2 && !/^(no|number|date)$|(?<!\p{L})(nom|names?|surname|given|pr[eé]noms?|birth|place)(?!\p{L})/iu.test(v) ? v : null;
};
/** a document number: letters and digits with at least one digit, so a word such as "IDENTITY" is never taken for one */
const codeValue = (min: number) => (s: string) => (s.replace(/^[\s:./-]+/, '').match(new RegExp('\\b[A-Z0-9][A-Z0-9-]{' + (min - 1) + ',}\\b', 'g')) ?? []).find((v) => /\d/.test(v)) ?? null;

const COMPARED = /number|^Full name$/i;
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;
const sameValue = (a: string, b: string) => a.toUpperCase().replace(/[^A-Z0-9]/g, '') === b.toUpperCase().replace(/[^A-Z0-9]/g, '');

/**
 * Key facts of an identity document. `second` is the same page read by the other reader
 * (the neural one). Facts are worked out from each reading separately and compared:
 * a value both readers give is marked as seen twice, a value only one found is used, and
 * where they differ the reading of `prefer` is kept and the other is remembered as `alt`.
 * Values proven by a check digit are never replaced.
 */
export function extractFacts(type: DocType, words: OcrWord[], fields: Field[], mrz: MrzResult | null, second: OcrWord[] = [], prefer: 'first' | 'second' = 'second'): Fact[] {
  const a = factsFrom(type, words, fields, mrz, second);
  if (!second.length || type === 'form' || type === 'other') return a;
  const b = factsFrom(type, second, [], mrz, words);
  const out = a.map((f): Fact => {
    if (f.from === 'mrz') return f;
    const g = b.find((x) => x.key === f.key);
    if (!g || g.from === 'mrz') return f;
    if (sameValue(f.value, g.value)) return { ...f, twice: true };
    // Two different dates under one label: a reader that lost the label picks up a neighbouring date.
    // The birth date is the earliest date on a document and the expiry the latest.
    const isDate = ISO_DATE.test(f.value) && ISO_DATE.test(g.value);
    // free text (a place, a nationality) is left to the first reader: only numbers, names and dates were measured
    if (!isDate && !COMPARED.test(f.key)) return f;
    const useSecond = isDate ? (f.key === 'Date of birth' ? g.value < f.value : f.key === 'Expiry date' ? g.value > f.value : false) : prefer === 'second';
    return useSecond ? { ...g, alt: f.value } : { ...f, alt: g.value };
  });
  for (const g of b) if ((COMPARED.test(g.key) || ISO_DATE.test(g.value)) && !out.some((f) => f.key === g.key)) out.push(g);
  return out;
}

function factsFrom(type: DocType, words: OcrWord[], fields: Field[], mrz: MrzResult | null, also: OcrWord[]): Fact[] {
  const lines = linesOf(words);
  const facts: Fact[] = [];
  const have = (k: string) => facts.some((f) => f.key === k);
  const add = (f: Fact) => { if (f.value && !have(f.key)) facts.push(f); };

  if (mrz) {
    const g = (k: string) => mrz.fields.find((f) => f.key === k);
    const name = [g('Given names')?.value, g('Surname')?.value].filter(Boolean).join(' ');
    const mended = mendName(name, also.length ? [...words, ...also] : words);
    add({ key: 'Full name', value: mended.name, from: 'mrz', twice: mended.twice });
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
    if (!have('Full name')) {
      // both labels on one row and both values on the row below: split the values where the second label starts
      const rows = rowsOf(words);
      for (let i = 0; i + 1 < rows.length && !have('Full name'); i++) {
        const near = (w: OcrWord, t: string) => { const l = lettersOnly(w.text); return l.length >= t.length - 1 && editDistance(l, t) <= 2; };
        const s = rows[i].find((w) => near(w, 'surname')), g = rows[i].find((w) => near(w, 'given') || near(w, 'first'));
        if (!s || !g || s === g) continue;
        const vals = rows[i + 1].filter((w) => /^[\p{L}'\-]{2,}$/u.test(w.text));
        const cut = Math.max(s.box[0], g.box[0]) - (g.box[3] - g.box[1]) * 0.6;
        const left = vals.filter((w) => (w.box[0] + w.box[2]) / 2 < cut).map((w) => w.text).join(' '), right = vals.filter((w) => (w.box[0] + w.box[2]) / 2 >= cut).map((w) => w.text).join(' ');
        if (left && right && nameValue(left) && nameValue(right)) add({ key: 'Full name', value: s.box[0] < g.box[0] ? `${right} ${left}` : `${left} ${right}`, from: 'text' });
      }
    }
    const sur = labelled(lines, fields, /surname|family\s+name|apellidos?|sobrenome|\bnom\b/i, nameValue);
    const giv = labelled(lines, fields, /given\s+names?|first\s+names?|pr[eé]noms?|nombres?/i, nameValue);
    if (sur && giv && lettersOnly(sur) !== lettersOnly(giv)) add({ key: 'Full name', value: `${giv} ${sur}`, from: 'text' });
    const name = labelled(lines, fields, /^\s*(full\s+)?name\b|\bname\s*:/i, textValue);
    if (name) add({ key: 'Full name', value: name, from: 'text' });
    if (!have('Full name')) {
      // labels with a letter or two misread ("Sarmame", "Ful Mame")
      const fs = fuzzyLabelled(lines, ['surname', 'family name'], nameValue), fg = fuzzyLabelled(lines, ['given names', 'given name', 'first name', 'first names'], nameValue);
      if (fs && fg && lettersOnly(fs) !== lettersOnly(fg)) add({ key: 'Full name', value: `${fg} ${fs}`, from: 'text' });
      const fn = fuzzyLabelled(lines, ['full name', 'name'], nameValue);
      if (fn) add({ key: 'Full name', value: fn, from: 'text' });
    }
    const nat = labelled(lines, fields, /nationality/i, (s) => s.match(/[A-Za-z][A-Za-z ]{2,}/)?.[0].trim() ?? null);
    if (nat) add({ key: 'Nationality', value: nat, from: 'text' });
    const dob = labelled(lines, fields, /date\s+of\s+birth|\bDOB\b|birth\s+date/i, parseDate);
    if (dob) add({ key: 'Date of birth', value: dob, from: 'text' });
    const iss = labelled(lines, fields, /date\s+of\s+issue|issue\s+date|issuing\s+date/i, parseDate);
    if (iss) add({ key: 'Issue date', value: iss, from: 'text' });
    const exp = labelled(lines, fields, /date\s+of\s+expiry|expiry(\s+date)?|expiration|valid\s+until|\buntil\b|expires/i, parseDate);
    if (exp) add({ key: 'Expiry date', value: exp, from: 'text' });
    if (!have('Date of birth')) { const d = fuzzyLabelled(lines, ['date of birth', 'birth date'], parseDate); if (d) add({ key: 'Date of birth', value: d, from: 'text' }); }
    if (!have('Issue date')) { const d = fuzzyLabelled(lines, ['date of issue', 'issue date'], parseDate); if (d) add({ key: 'Issue date', value: d, from: 'text' }); }
    if (!have('Expiry date')) { const d = fuzzyLabelled(lines, ['date of expiry', 'expiry date', 'valid until'], parseDate); if (d) add({ key: 'Expiry date', value: d, from: 'text' }); }
  }
  if ((type === 'id_card' || type === 'emirates_id' || type === 'driving_licence') && !have('Full name')) {
    // No label was read at all. Near the top of a card, under the heading, the holder's name stands on
    // a row of its own in capitals. Two blocks set far apart are the surname column and the given-name column.
    const top = rowsOf(words).slice(0, 8);
    const row = top.map((r) => r.filter((w) => /^\p{Lu}[\p{Lu}'\-]{2,}$/u.test(w.text) && w.conf >= 60)).find((r, i) =>
      i > 0 && r.length >= 2 && r.length <= 4 && !top[i].some((w) => /\d/.test(w.text)) && !HEADER_WORDS.test(r.map((w) => w.text).join(' ')));
    if (row) {
      const h = Math.max(...row.map((w) => w.box[3] - w.box[1]));
      let gi = -1, gap = 0;
      for (let i = 1; i < row.length; i++) { const d = row[i].box[0] - row[i - 1].box[2]; if (d > gap) { gap = d; gi = i; } }
      const t = (ws: OcrWord[]) => ws.map((w) => w.text).join(' ');
      add({ key: 'Full name', value: gap > h * 2.5 ? `${t(row.slice(gi))} ${t(row.slice(0, gi))}` : t(row), from: 'text' });
    }
  }
  if (type === 'id_card' && !have('Document number')) {
    // a card number standing unlabelled: mostly digits, not a date, sometimes with a letter or two in front
    const tok = lines.filter((l) => !parseDate(l)).flatMap((l) => l.replace(/\$/g, 'S').match(/\b[A-Z]{0,3}\d{6,10}\b/g) ?? [])[0];
    if (tok) add({ key: 'Document number', value: tok, from: 'text' });
  }
  if (type !== 'form' && type !== 'other' && type !== 'vehicle_licence') {
    // Dates nobody labelled: on an identity document the earliest date is the birth date and the
    // latest is the expiry, with the issue date between them. Used only for dates still missing.
    const dates = [...new Set(lines.filter((l) => !CONVENTION.test(l)).map(parseDate).filter((d): d is string => !!d))].sort();
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
    const plain = (l?: string) => { const v = l ? nameValue(l.replace(/^\d[ab]?[.,]?\s+/, '')) : null; return v && /^[\p{Lu} '\-]+$/u.test(v) ? v : null; };
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

export function alertsFor(type: DocType, typeConfidence: number, facts: Fact[], mrz: MrzResult | null, today = new Date(), quality?: { blurry: boolean; glary: boolean }): Alert[] {
  const out: Alert[] = [];
  // a photo that cannot be read properly comes before anything read from it
  if (quality?.blurry) out.push({ level: 'bad', text: 'This photo is blurry. Take it again, holding the phone still' });
  // glare is only called on cards and booklets: on a sheet of white paper a bright patch is just bright paper
  if (quality?.glary && type !== 'form' && type !== 'other') out.push({ level: quality.blurry ? 'warn' : 'bad', text: 'A light is reflecting off the page. Tilt it and take the photo again' });
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

/** The printed labels an identity document can carry, as they should be shown. */
const ID_LABELS = ['Name', 'Surname', 'Given names', 'Nationality', 'Date of birth', 'Place of birth', 'Sex', 'Issue date', 'Date of issue', 'Expiry date', 'Date of expiry', 'Valid until', 'Place of issue', 'Issuing authority', 'Licensing authority', 'Authority', 'Licence no', 'License no', 'Passport no', 'Document no', 'Card number', 'ID number', 'Occupation', 'Profession', 'Employer', 'Sponsor', 'Blood group', 'Class', 'Category', 'Address', 'File no', 'UID no', 'Entry permit no', 'Visa type', 'Duration of stay', 'Plate no', 'Owner', 'Model', 'Colour', 'Chassis no', 'Engine no', 'Insurance', 'Traffic code no', 'Vehicle type', 'Origin', 'Type'];
const letters = (v: string) => v.toUpperCase().replace(/[^A-Z]/g, '');
/**
 * On an identity document the label-and-value pairs found by the form models are mostly noise
 * (background patterns, half a word of Arabic, a hologram). Keep a pair only when its label is
 * one a document really prints and its value looks like text; show the label spelled properly;
 * and drop a pair that only repeats a key fact. What is left is short and readable, or empty.
 */
export function tidyIdFields<T extends Field>(fields: T[], facts: Fact[]): T[] {
  const flat = (v: string) => v.toUpperCase().replace(/[^A-Z0-9]/g, '');
  const known = new Set(facts.map((f) => flat(f.value)));
  const out: T[] = [];
  for (const f of fields) {
    const v = f.value.trim(), k = letters(f.key);
    if (!v || k.length < 3 || f.confidence < 0.5) continue;
    const clean = v.replace(/[^A-Za-z0-9 .,\/-]/g, '');
    if (flat(v).length < 2 || clean.length / v.length < 0.85) continue;
    let best = '', bestD = 1;
    for (const l of ID_LABELS) { const t = letters(l); const d = editDistance(k, t) / Math.max(k.length, t.length); if (d < bestD) { bestD = d; best = l; } }
    if (bestD > 0.3) continue;
    if (known.has(flat(v)) || out.some((o) => o.key === best)) continue;
    out.push({ ...f, key: best });
  }
  return out;
}
