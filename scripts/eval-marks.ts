// Checkbox and signature detection on the 20-form test set.
//
// Truth comes from the PDFs: every checkbox widget on page 1 (about a third were ticked
// by make-forms20.py) and every signing line (about half carry a synthetic pen scribble).
//
//   checkboxes, free     find the boxes on the page with no prior knowledge, then read them
//   checkboxes, template the template says where each box is; only its state is read
//   signatures           find each "Signature" caption and decide: signed or not
//
// usage: npx tsx scripts/eval-marks.ts     (run eval-templates.ts first: it writes the library)
import fs from 'node:fs';
import path from 'node:path';
import type { Box } from '../src/core/types';
import { checkState, dropLetterBoxes, findCheckboxes, findSignatures, isSignatureCaption } from '../src/core/marks';
import { matchTemplate, templateChecks, templateWords, type Template } from '../src/core/template';
import { ocrAll, pageGray, ROOT } from './forms20-ocr';

const SET = path.join(ROOT, 'testsets/forms20');
const truth = JSON.parse(fs.readFileSync(path.join(SET, 'truth.json'), 'utf8'));
const files: string[] = truth.forms.map((f: any) => f.file);
const library: Template[] = JSON.parse(fs.readFileSync(path.join(ROOT, 'public/templates/library.json'), 'utf8')).templates;
const ocr = await ocrAll('tesseract', SET, files, 'ocr');

const cx = (b: Box) => (b[0] + b[2]) / 2, cy = (b: Box) => (b[1] + b[3]) / 2;
type G = { boxes: number; found: number; detections: number; matchedDet: number; stateOk: number; ticked: number; tickedOk: number; tplBoxes: number; tplOk: number; tplTicked: number; tplTickedOk: number; tplFalseTick: number; tplSigLines: number; tplSigOk: number; tplSignedFound: number; tplFalseSigned: number; sigLines: number; sigCaption: number; sigOk: number; signed: number; signedFound: number; falseSigned: number; ms: number; forms: number };
const zero = (): G => ({ boxes: 0, found: 0, detections: 0, matchedDet: 0, stateOk: 0, ticked: 0, tickedOk: 0, tplBoxes: 0, tplOk: 0, tplTicked: 0, tplTickedOk: 0, tplFalseTick: 0, tplSigLines: 0, tplSigOk: 0, tplSignedFound: 0, tplFalseSigned: 0, sigLines: 0, sigCaption: 0, sigOk: 0, signed: 0, signedFound: 0, falseSigned: 0, ms: 0, forms: 0 });
const groups: Record<string, G> = { all: zero(), clean: zero(), 'scan-like': zero() };
const perForm: any[] = [];

for (const f of truth.forms) {
  const { g, scale } = await pageGray(path.join(SET, f.file));
  const o = ocr.get(f.file)!;
  const t = zero();
  t.forms = 1;
  const t0 = performance.now();
  const det = dropLetterBoxes(findCheckboxes(g), o.words, scale);
  const sigs = findSignatures(g, o.words, scale);
  t.ms = performance.now() - t0;

  // --- checkboxes, free detection ---
  const used = new Set<number>();
  t.detections = det.length;
  for (const c of f.checkboxes as { box: Box; checked: boolean }[]) {
    const b = c.box.map((v) => v * scale) as Box;
    const size = Math.max(b[2] - b[0], b[3] - b[1]);
    t.boxes++;
    if (c.checked) t.ticked++;
    let best = -1, bd = size * 0.9;
    det.forEach((d, i) => { const dist = Math.hypot(cx(d.box) - cx(b), cy(d.box) - cy(b)); if (!used.has(i) && dist < bd) { bd = dist; best = i; } });
    if (best < 0) continue;
    used.add(best);
    t.found++;
    if (det[best].checked === c.checked) t.stateOk++;
    if (c.checked && det[best].checked) t.tickedOk++;
  }
  t.matchedDet = used.size;

  // --- checkboxes, template ---
  const m = matchTemplate(library, o.words, o.width, o.height);
  if (m && m.template.checks?.length && m.template.id === f.file.replace(/\.\w+$/, '')) {
    const places = templateChecks(m, o.width, o.height);
    (f.checkboxes as { checked: boolean }[]).forEach((c, i) => {
      const b = places[i].box.map((v) => v * scale) as Box;
      // the alignment can be a few pixels off: snap to a detected outline when one is right there
      const size = Math.max(b[2] - b[0], b[3] - b[1]);
      const near = det.find((d) => Math.hypot(cx(d.box) - cx(b), cy(d.box) - cy(b)) < size * 0.6);
      const got = near ? near.checked : checkState(g, b).checked;
      t.tplBoxes++;
      if (got === c.checked) t.tplOk++;
      if (c.checked) { t.tplTicked++; if (got) t.tplTickedOk++; } else if (got) t.tplFalseTick++;
    });
  }

  // --- signatures ---
  for (const s of f.signatures as { caption: Box; signed: boolean }[]) {
    t.sigLines++;
    if (s.signed) t.signed++;
    const h = s.caption[3] - s.caption[1];
    const d = sigs.find((x) => Math.abs(x.caption[1] - s.caption[1]) < h * 2.5 && Math.abs(x.caption[0] - s.caption[0]) < h * 6);
    if (process.env.DBG && (!d || d.signed !== s.signed)) console.log('   sig miss', f.file, f.quality, 'truth signed', s.signed, d ? `detected signed ${d.signed} ink ${d.ink}` : 'caption not found', JSON.stringify(s.caption));
    if (!d) continue;
    t.sigCaption++;
    if (d.signed === s.signed) t.sigOk++;
    if (s.signed && d.signed) t.signedFound++;
    if (!s.signed && d.signed) t.falseSigned++;
  }
  // --- signatures, template: the caption places come from the blank form ---
  if (m && m.template.id === f.file.replace(/\.\w+$/, '')) {
    const tsigs = findSignatures(g, o.words, scale, templateWords(m, o.width, o.height, isSignatureCaption));
    for (const s of f.signatures as { caption: Box; signed: boolean }[]) {
      const h = s.caption[3] - s.caption[1];
      const d = tsigs.find((x) => Math.abs(x.caption[1] - s.caption[1]) < h * 2.5 && Math.abs(x.caption[0] - s.caption[0]) < h * 6);
      t.tplSigLines++;
      const got = d?.signed ?? false;
      if (got === s.signed) t.tplSigOk++;
      if (s.signed && got) t.tplSignedFound++;
      if (!s.signed && got) t.tplFalseSigned++;
    }
  }
  for (const k of ['all', f.quality]) for (const key of Object.keys(t) as (keyof G)[]) groups[k][key] += t[key];
  perForm.push({ file: f.file, quality: f.quality, checkboxes: t.boxes, found: t.found, stateOk: t.stateOk, extraDetections: t.detections - t.matchedDet, templateOk: t.tplOk, signatureLines: t.sigLines, signatureOk: t.sigOk });
}

const d = (a: number, b: number) => (b ? a / b : 0);
const rate = (t: G) => ({
  forms: t.forms,
  checkboxes: { boxes: t.boxes, ticked: t.ticked, recall: d(t.found, t.boxes), precision: d(t.matchedDet, t.detections), stateAccuracy: d(t.stateOk, t.found), tickedRecall: d(t.tickedOk, t.ticked), endToEnd: d(t.stateOk, t.boxes) },
  checkboxesTemplate: { boxes: t.tplBoxes, stateAccuracy: d(t.tplOk, t.tplBoxes), tickedRecall: d(t.tplTickedOk, t.tplTicked), falseTicks: t.tplFalseTick },
  signatures: { lines: t.sigLines, signed: t.signed, captionFound: d(t.sigCaption, t.sigLines), accuracy: d(t.sigOk, t.sigLines), signedRecall: d(t.signedFound, t.signed), falseSigned: t.falseSigned },
  signaturesTemplate: { lines: t.tplSigLines, accuracy: d(t.tplSigOk, t.tplSigLines), signedRecall: d(t.tplSignedFound, t.signed), falseSigned: t.tplFalseSigned },
  msPerPage: t.ms / t.forms,
});
const R = { generated: new Date().toISOString(), note: 'Ticks come from the PDF form fields; signatures are synthetic pen scribbles, not real signatures.', all: rate(groups.all), clean: rate(groups.clean), scanLike: rate(groups['scan-like']), perForm };
const pc = (v: number) => (v * 100).toFixed(1) + '%';
for (const [k, t] of Object.entries(groups)) {
  const r = rate(t);
  console.log(`${k.padEnd(9)} checkboxes ${t.boxes}: found ${pc(r.checkboxes.recall)}, precision ${pc(r.checkboxes.precision)}, state right ${pc(r.checkboxes.stateAccuracy)}, end-to-end ${pc(r.checkboxes.endToEnd)} | template: state right ${pc(r.checkboxesTemplate.stateAccuracy)} of ${t.tplBoxes}, ticked found ${pc(r.checkboxesTemplate.tickedRecall)}, false ticks ${t.tplFalseTick}`);
  console.log(`${''.padEnd(9)} signatures ${t.sigLines} lines (${t.signed} signed): caption found ${pc(r.signatures.captionFound)}, right ${pc(r.signatures.accuracy)}, signed found ${pc(r.signatures.signedRecall)}, false signed ${t.falseSigned} | template: right ${pc(r.signaturesTemplate.accuracy)}, signed found ${pc(r.signaturesTemplate.signedRecall)}, false signed ${t.tplFalseSigned}   ${r.msPerPage.toFixed(0)} ms/page`);
}
for (const p of perForm) if (p.found < p.checkboxes * 0.7 || p.extraDetections > 3 || p.signatureOk < p.signatureLines) console.log('  ', JSON.stringify(p));
const resPath = path.join(ROOT, 'public/eval/results.json');
const all = JSON.parse(fs.readFileSync(resPath, 'utf8'));
all.marks = R;
fs.writeFileSync(resPath, JSON.stringify(all, null, 1));
console.log('wrote the marks block of public/eval/results.json');
