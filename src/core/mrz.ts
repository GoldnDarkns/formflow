// ICAO Doc 9303 machine-readable zone (MRZ) parser for passports (TD3),
// ID cards such as the Emirates ID (TD1), TD2 documents and visas (MRV-A/B).
// Check digits give a hard, model-independent validity signal for each field,
// and are used to repair common OCR confusions (O/0, I/1, S/5, B/8 ...).

export interface MrzField {
  key: string;
  value: string;
  valid?: boolean; // check digit result when the field has one
}

export interface MrzResult {
  format: 'TD1' | 'TD2' | 'TD3' | 'MRVA' | 'MRVB';
  docType: string;
  lines: string[];
  fields: MrzField[];
  valid: boolean; // all check digits pass
  checks: { name: string; ok: boolean }[];
}

const CHAR_VAL = (c: string) => (c === '<' ? 0 : /\d/.test(c) ? +c : c.charCodeAt(0) - 55);
export function checkDigit(s: string): number {
  const w = [7, 3, 1];
  let sum = 0;
  for (let i = 0; i < s.length; i++) sum += CHAR_VAL(s[i]) * w[i % 3];
  return sum % 10;
}
const checkOk = (data: string, d: string) => /\d/.test(d) && checkDigit(data) === +d;

const TO_DIGIT: Record<string, string> = { O: '0', Q: '0', D: '0', U: '0', I: '1', L: '1', T: '1', Z: '2', S: '5', B: '8', G: '6', A: '4' };
const TO_ALPHA: Record<string, string> = { '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A', '7': 'Z' };
// characters OCR commonly confuses in OCR-B, used when a check digit fails
const CONFUSE: Record<string, string[]> = {
  '0': ['O', 'D', 'Q'], O: ['0', 'D', 'Q'], D: ['0', 'O'], Q: ['0', 'O'], '1': ['I', 'L', 'T'], I: ['1', 'L'], L: ['1', 'I'], T: ['1', '7'],
  '2': ['Z'], Z: ['2', '7'], '7': ['Z', 'T', '1'], '5': ['S'], S: ['5'], '8': ['B'], B: ['8'], '6': ['G'], G: ['6'], '4': ['A'], A: ['4'],
};
const digits = (s: string) => s.replace(/[A-Z]/g, (c) => TO_DIGIT[c] ?? c);
const alpha = (s: string) => s.replace(/\d/g, (c) => TO_ALPHA[c] ?? c);

/** Try swapping ambiguous characters until the check digit passes (alphanumeric doc numbers). */
function repair(data: string, d: string): { data: string; d: string; ok: boolean; repaired?: boolean } {
  const dd = digits(d);
  if (checkOk(data, dd)) return { data, d: dd, ok: true };
  // single substitutions first (most likely), then pairs
  const amb: number[] = [];
  for (let i = 0; i < data.length; i++) if (CONFUSE[data[i]]) amb.push(i);
  for (const p of amb)
    for (const c of CONFUSE[data[p]]) {
      const s = data.slice(0, p) + c + data.slice(p + 1);
      if (checkOk(s, dd)) return { data: s, d: dd, ok: true, repaired: true };
    }
  for (let x = 0; x < amb.length; x++)
    for (let y = x + 1; y < amb.length; y++)
      for (const c1 of CONFUSE[data[amb[x]]])
        for (const c2 of CONFUSE[data[amb[y]]]) {
          const a = data.split('');
          a[amb[x]] = c1;
          a[amb[y]] = c2;
          const s = a.join('');
          if (checkOk(s, dd)) return { data: s, d: dd, ok: true, repaired: true };
        }
  return { data, d: dd, ok: false };
}

/** normalise one OCR'd line into MRZ alphabet */
export function cleanMrzLine(s: string): string {
  return s
    .toUpperCase()
    .replace(/[«‹(\[{]/g, '<')
    .replace(/\s+/g, '')
    .replace(/[^A-Z0-9<]/g, '');
}

/** Find MRZ-looking lines in free OCR text. */
export function findMrzLines(text: string): string[] {
  // the first line of an ID card (TD1) can have no filler characters at all
  return text.split(/\n/).map(cleanMrzLine).filter((l) => l.length >= 26);
}

function fit(l: string, n: number) {
  return l.length >= n ? l.slice(0, n) : l + '<'.repeat(n - l.length);
}

function names(s: string) {
  // the "<" padding after a name is often misread as a run of K and L; no real name has four of those in a row
  // (a K or L that ends a real name, as in RAFAEL<<<<, sits before the first "<" and is kept)
  const line = alpha(s).replace(/[KL<]{4,}/g, (m) => { const i = m.indexOf('<'); return i < 0 || i > 2 ? '<'.repeat(m.length) : m.slice(0, i) + '<'.repeat(m.length - i); });
  const [sur, given = ''] = line.split('<<');
  return { surname: sur.replace(/</g, ' ').trim(), given: given.replace(/</g, ' ').replace(/\s+/g, ' ').trim() };
}

function date(yymmdd: string, future: boolean): string {
  if (!/^\d{6}$/.test(yymmdd)) return yymmdd;
  const yy = +yymmdd.slice(0, 2);
  const now = new Date().getFullYear() % 100;
  const century = future ? (yy < now + 50 ? 2000 : 1900) : yy > now ? 1900 : 2000;
  return `${century + yy}-${yymmdd.slice(2, 4)}-${yymmdd.slice(4, 6)}`;
}

/** A repair that the composite check digit does not confirm is a guess: show the raw reading, unverified. */
function unverify(fields: MrzField[], checks: { name: string; ok: boolean }[], key: string, raw: string) {
  const f = fields.find((x) => x.key === key);
  if (f) { f.value = raw.replace(/<+$/g, '').replace(/</g, ' ').trim(); f.valid = false; }
  const c = checks.find((x) => x.name === key);
  if (c) c.ok = false;
}

const SEX: Record<string, string> = { M: 'Male', F: 'Female', '<': 'Unspecified', X: 'Unspecified' };
const strip = (s: string) => s.replace(/<+$/g, '').replace(/</g, ' ').trim();

export function parseMrz(rawLines: string[]): MrzResult | null {
  const lines = rawLines.map(cleanMrzLine).filter((l) => l.length >= 26);
  if (lines.length < 2) return null;
  // choose layout by line lengths
  let fmt: MrzResult['format'];
  const L = lines.slice(-3);
  // three short lines, the first starting with the ID-card type code (I, A or C)
  if (L.length === 3 && /^[IAC][A-Z<]/.test(L[0]) && L.every((l) => l.length <= 38) && !/^P</.test(L[1])) fmt = 'TD1';
  else {
    const two = lines.slice(-2);
    const len = Math.round((two[0].length + two[1].length) / 2);
    const isVisa = two[0][0] === 'V';
    if (len >= 40) fmt = isVisa ? 'MRVA' : 'TD3';
    else fmt = isVisa ? 'MRVB' : 'TD2';
  }
  const checks: { name: string; ok: boolean }[] = [];
  const fields: MrzField[] = [];
  const push = (key: string, value: string, valid?: boolean) => fields.push({ key, value, valid });

  if (fmt === 'TD1') {
    const [a, b, c] = L.map((l) => fit(l, 30));
    const docType = strip(alpha(a.slice(0, 2)));
    const country = alpha(a.slice(2, 5));
    const num = repair(a.slice(5, 14), a[14]);
    const optional1 = a.slice(15, 30);
    const dob = digits(b.slice(0, 6)), dobOk = checkOk(dob, digits(b[6]));
    const exp = digits(b.slice(8, 14)), expOk = checkOk(exp, digits(b[14]));
    const nat = alpha(b.slice(15, 18));
    const optional2 = b.slice(18, 29);
    const composite = num.data + num.d + optional1 + dob + digits(b[6]) + exp + digits(b[14]) + optional2;
    const compOk = checkOk(composite, digits(b[29]));
    const n = names(c);
    checks.push({ name: 'Document number', ok: num.ok }, { name: 'Date of birth', ok: dobOk }, { name: 'Expiry date', ok: expOk }, { name: 'Composite', ok: compOk });
    push('Document type', docType);
    push('Issuing country', country);
    push('Document number', strip(num.data), num.ok);
    if (strip(optional1)) push('Optional data', strip(optional1));
    push('Surname', n.surname);
    push('Given names', n.given);
    push('Nationality', nat);
    push('Date of birth', date(dob, false), dobOk);
    push('Sex', SEX[b[7]] ?? b[7]);
    push('Expiry date', date(exp, true), expOk);
    if (strip(optional2)) push('Optional data 2', strip(optional2));
    if (num.repaired && !compOk) unverify(fields, checks, 'Document number', a.slice(5, 14));
    return { format: fmt, docType, lines: [a, b, c], fields, checks, valid: checks.every((c) => c.ok) };
  }

  const n = fmt === 'TD3' || fmt === 'MRVA' ? 44 : 36;
  const [a, b] = lines.slice(-2).map((l) => fit(l, n));
  const docType = strip(alpha(a.slice(0, 2)));
  const country = alpha(a.slice(2, 5));
  const nm = names(a.slice(5));
  const num = repair(b.slice(0, 9), b[9]);
  const nat = alpha(b.slice(10, 13));
  const dob = digits(b.slice(13, 19)), dobOk = checkOk(dob, digits(b[19]));
  const sex = b[20];
  const exp = digits(b.slice(21, 27)), expOk = checkOk(exp, digits(b[27]));
  checks.push({ name: 'Document number', ok: num.ok }, { name: 'Date of birth', ok: dobOk }, { name: 'Expiry date', ok: expOk });
  push('Document type', docType);
  push('Issuing country', country);
  push('Surname', nm.surname);
  push('Given names', nm.given);
  push('Document number', strip(num.data), num.ok);
  push('Nationality', nat);
  push('Date of birth', date(dob, false), dobOk);
  push('Sex', SEX[sex] ?? sex);
  push('Expiry date', date(exp, true), expOk);
  if (fmt === 'TD3') {
    let pn = b.slice(28, 42), pnD = b[42];
    let pnOk: boolean;
    if (pn.replace(/</g, '') === '') pnOk = pnD === '<' || pnD === '0';
    else {
      const r = repair(pn, pnD);
      pn = r.data;
      pnD = r.d;
      pnOk = r.ok;
    }
    const composite = num.data + num.d + dob + digits(b[19]) + exp + digits(b[27]) + pn + (pnD === '<' ? '<' : digits(pnD));
    const compOk = checkOk(composite, digits(b[43]));
    if (strip(pn)) push('Personal number', strip(pn), pnOk);
    checks.push({ name: 'Personal number', ok: pnOk }, { name: 'Composite', ok: compOk });
    if (num.repaired && !compOk) unverify(fields, checks, 'Document number', b.slice(0, 9));
  } else if (fmt === 'TD2') {
    const opt = b.slice(28, 35);
    const composite = num.data + num.d + dob + digits(b[19]) + exp + digits(b[27]) + opt;
    const compOk = checkOk(composite, digits(b[35]));
    if (strip(opt)) push('Optional data', strip(opt));
    checks.push({ name: 'Composite', ok: compOk });
  } else {
    const opt = b.slice(28, n);
    if (strip(opt)) push('Optional data', strip(opt));
  }
  return { format: fmt, docType, lines: [a, b], fields, checks, valid: checks.every((c) => c.ok) };
}
