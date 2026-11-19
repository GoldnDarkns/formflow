import type { DocResult } from '../engine/engine';
import type { Analysis } from '../core/pipeline';
import type { Field } from '../core/types';
import { similarity } from '../core/ocr';
import { DOC_LABEL, EID_RE } from '../core/classify';
import type { VisionOpinion } from '../core/critic-vision';
import { critique } from '../core/critic';
import { alertsFor } from '../core/facts';

export interface EField extends Field {
  uid: string;
  edited?: boolean;
  confirmed?: boolean; // a person checked this field
  mrzMatch?: string; // name of the MRZ field it matches
}

export interface DocState {
  doc: DocResult;
  analysis: Analysis;
  fields: EField[];
  second: 'none' | 'running' | 'done' | 'error';
  vision?: VisionOpinion | null; // the vision critic's opinion of the type
  secondError?: string;
}

let n = 0;
export function toEditable(fields: Field[], prev?: EField[]): EField[] {
  const out = fields.map((f) => ({ ...f, uid: `f${++n}` }) as EField);
  if (prev) {
    // keep a person's edits and confirmations when the analysis is refreshed
    for (const p of prev) {
      if (!p.edited && !p.confirmed) continue;
      const m = out.find((o) => o.qId === p.qId && p.qId !== undefined);
      if (m) Object.assign(m, { key: p.key, value: p.value, edited: p.edited, confirmed: p.confirmed, status: p.confirmed ? 'ok' : m.status });
      else out.push(p);
    }
  }
  return out;
}

/** Cross-check visual-zone fields against the MRZ (passport number, names). */
export function markMrz(state: DocState): EField[] {
  const mrz = state.doc.mrz;
  if (!mrz) return state.fields;
  const targets = mrz.fields.filter((f) => ['Document number', 'Surname', 'Given names'].includes(f.key) && f.value);
  return state.fields.map((f) => {
    const t = targets.find((t) => f.value && similarity(f.value.replace(/\s/g, ''), t.value.replace(/\s/g, '')) >= 0.85);
    if (!t) return f;
    // the same value appears in the MRZ and its check digits pass: verified by arithmetic
    return mrz.valid ? { ...f, mrzMatch: t.key, confidence: Math.max(f.confidence, 0.99), status: 'ok' as const, reasons: [] } : { ...f, mrzMatch: t.key };
  });
}

export function needsReview(fs: EField[]) {
  return fs.filter((f) => !f.confirmed && f.status === 'review').length;
}

export function exportJson(s: DocState) {
  const d = s.doc;
  const trust = trustOf(s);
  return {
    document: d.fileName,
    documentType: d.docType,
    documentTypeConfidence: Math.round(d.classification.confidence * 100) / 100,
    trustScore: Math.round(trust.score * 100) / 100,
    keyFacts: Object.fromEntries(d.facts.map((f) => [f.key, f.value])),
    alerts: alertsFor(d.docType, d.classification.confidence, d.facts, d.mrz).filter((a) => a.level !== 'good').map((a) => a.text),
    processedAt: new Date().toISOString(),
    engine: { ocr: 'tesseract.js (eng)', layout: 'FormFlow GBDT (ONNX)', secondReader: s.second === 'done' ? 'Florence-2-base-ft' : null },
    mrz: d.mrz ? { format: d.mrz.format, valid: d.mrz.valid, lines: d.mrz.lines, fields: Object.fromEntries(d.mrz.fields.map((f) => [f.key, f.value])), checks: d.mrz.checks } : null,
    fields: s.fields.map((f) => ({
      key: f.key,
      value: f.value,
      confidence: Math.round(f.confidence * 100) / 100,
      status: f.confirmed ? 'confirmed' : f.status,
      editedByHuman: !!f.edited,
      ...(f.alt ? { secondReaderValue: f.alt } : {}),
      ...(f.mrzMatch ? { matchesMrz: f.mrzMatch } : {}),
    })),
  };
}

const csvCell = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
export function exportCsv(states: DocState[]) {
  const rows = [['document', 'source', 'key', 'value', 'confidence', 'status', 'edited_by_human']];
  for (const s of states) {
    rows.push([s.doc.fileName, 'type', 'Document type', DOC_LABEL[s.doc.docType], s.doc.classification.confidence.toFixed(2), 'ok', 'false']);
    for (const f of s.doc.facts) rows.push([s.doc.fileName, 'key_fact', f.key, f.value, f.valid === false ? '0' : f.proof ? '1' : '', f.valid === false ? 'check_failed' : 'ok', 'false']);
    for (const f of s.doc.mrz?.fields ?? []) rows.push([s.doc.fileName, 'mrz', f.key, f.value, f.valid === false ? '0' : '1', f.valid === false ? 'check_failed' : 'ok', 'false']);
    for (const f of s.fields) rows.push([s.doc.fileName, 'layout', f.key, f.value, f.confidence.toFixed(2), f.confirmed ? 'confirmed' : f.status, String(!!f.edited)]);
  }
  return rows.map((r) => r.map(csvCell).join(',')).join('\n');
}

export function download(name: string, data: string, type: string) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Export the corrected document as a labelled example in FUNSD format, so a
 * clerk's corrections become test / training data (scripts/eval-own.ts).
 */
export function exportFunsd(s: DocState) {
  const ents = s.analysis.entities;
  const label = new Map<number, string>();
  const text = new Map<number, string>();
  const linking: [number, number][] = [];
  for (const f of s.fields) {
    if (f.qId === undefined) continue;
    label.set(f.qId, 'question');
    if (f.edited) text.set(f.qId, f.key);
    const as = s.analysis.links.filter((l) => l.q === f.qId).map((l) => l.a);
    if (f.aId !== undefined && !as.includes(f.aId)) as.push(f.aId);
    as.forEach((a, i) => {
      label.set(a, 'answer');
      linking.push([f.qId!, a]);
      if (f.edited && i === 0) text.set(a, f.value);
    });
  }
  return {
    source: s.doc.fileName,
    width: s.doc.width,
    height: s.doc.height,
    note: 'Coordinates are in the deskewed image frame used by FormFlow. Labels reflect the reviewer corrections.',
    form: ents.map((e) => ({
      id: e.id,
      box: e.box.map(Math.round),
      text: text.get(e.id) ?? e.text,
      label: label.get(e.id) ?? (e.label === 'header' ? 'header' : 'other'),
      words: e.words.map((w) => ({ box: w.box.map(Math.round), text: w.text })),
      linking: linking.filter(([q, a]) => q === e.id || a === e.id),
    })),
  };
}

// ---------------- trust score ----------------
export interface Trust {
  score: number; // 0..1
  parts: { label: string; value: number; note: string }[];
}
/**
 * One number per document: how far can the clerk trust this extraction?
 * Blends (a) the confidence of the extracted fields, (b) how sure the type
 * recognition is and (c) arithmetic checks (MRZ check digits, ID checksum).
 * Fields a person has confirmed count as certain.
 */
export function trustOf(s: DocState): Trust {
  const parts: (Trust['parts'][number] & { w: number })[] = [];
  const filled = s.fields.filter((f) => f.value);
  if (filled.length) {
    const v = filled.reduce((a, f) => a + (f.confirmed ? 1 : f.confidence), 0) / filled.length;
    parts.push({ label: 'Fields', value: v, w: 0.5, note: `${filled.length} fields, ${filled.filter((f) => f.confirmed).length} confirmed by you` });
  }
  parts.push({ label: 'Document type', value: s.doc.classification.confidence, w: 0.2, note: DOC_LABEL[s.doc.docType] });
  const checks: boolean[] = [];
  if (s.doc.mrz) checks.push(...s.doc.mrz.checks.map((c) => c.ok));
  for (const f of s.doc.facts) if (f.proof === 'checksum' && f.valid !== undefined) checks.push(f.valid);
  if (checks.length) parts.push({ label: 'Checksums', value: checks.filter(Boolean).length / checks.length, w: 0.3, note: `${checks.filter(Boolean).length} of ${checks.length} pass` });
  const W = parts.reduce((a, p) => a + p.w, 0);
  return { score: parts.reduce((a, p) => a + p.value * p.w, 0) / W, parts };
}

// ---------------- applicant packets: cross-document checks ----------------
export interface Packet {
  name: string;
  docs: DocState[];
  checks: { label: string; ok: boolean; detail: string }[];
}
const fact = (s: DocState, k: string) => s.doc.facts.find((f) => f.key === k)?.value ?? '';
const tokens = (n: string) => new Set(n.toUpperCase().replace(/[^A-Z ]/g, '').split(/\s+/).filter((t) => t.length > 1));
function samePerson(a: string, b: string) {
  const A = tokens(a), B = tokens(b);
  if (!A.size || !B.size) return false;
  const inter = [...A].filter((t) => B.has(t)).length;
  return inter / Math.min(A.size, B.size) >= 0.6 && inter >= 2;
}
export function packetsOf(docs: DocState[]): Packet[] {
  const groups: DocState[][] = [];
  for (const d of docs) {
    const n = fact(d, 'Full name');
    if (!n) continue;
    const g = groups.find((g) => samePerson(fact(g[0], 'Full name'), n));
    if (g) g.push(d);
    else groups.push([d]);
  }
  return groups.filter((g) => g.length > 1).map((g) => {
    const checks: Packet['checks'] = [];
    const compare = (label: string, key: string, types?: string[]) => {
      const vals = g.filter((d) => !types || types.includes(d.doc.docType)).map((d) => ({ v: fact(d, key).toUpperCase().replace(/\s+/g, ' ').trim(), t: DOC_LABEL[d.doc.docType] })).filter((x) => x.v);
      if (vals.length < 2) return;
      const ok = vals.every((x) => x.v === vals[0].v);
      checks.push({ label, ok, detail: ok ? `${vals[0].v} on ${vals.length} documents` : vals.map((x) => `${x.t}: ${x.v}`).join(' / ') });
    };
    compare('Date of birth', 'Date of birth');
    compare('Name spelling', 'Full name');
    compare('Passport number', 'Passport number', ['passport', 'visa']);
    // show the spelling most documents agree on
    const names = g.map((d) => fact(d, 'Full name'));
    const name = [...names].sort((a, b) => names.filter((n) => n === b).length - names.filter((n) => n === a).length)[0];
    return { name, docs: g, checks };
  });
}

export const critiqueOf = (s: DocState) => critique(s.doc.docType, s.doc.classification, s.doc.facts, s.doc.mrz, s.fields, s.doc.words, s.vision);

/** Text blocks that hold personal identifiers: hidden by default in a redacted copy. */
export function sensitiveEntities(s: DocState): Set<number> {
  const out = new Set<number>();
  const ids = s.doc.facts.filter((f) => /number|birth/i.test(f.key)).map((f) => f.value.replace(/\W/g, '').toUpperCase()).filter((v) => v.length >= 5);
  for (const e of s.analysis.entities) {
    const t = e.text.replace(/\s/g, '');
    if (/<{2,}/.test(t) || EID_RE.test(e.text) || ids.some((v) => t.replace(/\W/g, '').toUpperCase().includes(v))) out.add(e.id);
  }
  for (const f of s.fields)
    if (/\b(no|number|passport|birth|dob|licen[cs]e|uid|permit|id)\b/i.test(f.key)) for (const l of s.analysis.links) if (l.q === f.qId) out.add(l.a);
  return out;
}
