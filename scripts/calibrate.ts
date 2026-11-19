// Fit and test the confidence calibration.
//
// The raw field score is over-confident. We learn a monotone curve (isotonic
// regression) from raw score to "share of fields that were right" and measure
// the expected calibration error (ECE) before and after.
//
// Two protocols are reported:
//   train->test : curve fitted on the 149 FUNSD training forms, tested on the 50 test forms.
//                 The layout models were trained on those same training forms, so their
//                 probabilities are rosier there than on unseen pages.
//   cross-fit   : 5-fold over the 50 test forms (pages never seen by the layout models);
//                 each fold's curve is fitted on the other four folds.
//
// usage: npx tsx scripts/calibrate.ts [ship: cross|train]   (default cross)
import fs from 'node:fs';
import path from 'node:path';
import * as ortNode from 'onnxruntime-node';
import { analyse } from '../src/core/pipeline';
import { loadClassifiers } from '../src/core/classifier';
import { applyCurve, fitIsotonic, reliability, type Calibration, type Curve } from '../src/core/calibration';
import { loadGt, loadOcr, pages } from './funsd';
import { gtFields, matchFields } from './match';

const ROOT = path.resolve(import.meta.dirname, '..');
const ship = process.argv[2] ?? 'cross';
const clf = await loadClassifiers(ortNode as any, path.join(ROOT, 'public/models/entity_gbdt.onnx'), path.join(ROOT, 'public/models/link_gbdt.onnx'));

interface S { score: number; ok: boolean; page: number }
async function collect(split: 'training' | 'testing', mode: string, reader: 'tesseract' | 'ppocr'): Promise<S[]> {
  const out: S[] = [];
  const files = pages(split);
  for (let i = 0; i < files.length; i++) {
    const o = loadOcr(split, mode, files[i]);
    if (!o) continue;
    const an = await analyse(o.words, o.width, o.height, clf, undefined, 'form', reader);
    const m = matchFields(an.fields, gtFields(loadGt(split, files[i], o)));
    for (const r of m.res) out.push({ score: r.f.raw ?? r.f.confidence, ok: r.ok, page: i });
  }
  return out;
}
const rel = (s: S[], c?: Curve) => reliability(s.map((x) => ({ conf: c ? applyCurve(c, x.score) : x.score, ok: x.ok })));
const r3 = (v: number) => Math.round(v * 1000) / 1000;

const FOLDS = 5;
const cal: Calibration = { fitted: true, fittedOn: '', curves: {} };
const report: any = { generated: new Date().toISOString(), method: 'isotonic regression (pool adjacent violators), at most 12 knots, blocks of at least 25 fields', shipped: ship, readers: {} };

for (const [name, mode] of [['tesseract', 'full'], ['ppocr', 'pp']] as ['tesseract' | 'ppocr', string][]) {
  const train = await collect('training', mode, name);
  const test = await collect('testing', mode, name);
  if (!train.length || !test.length) { console.log(name, 'no OCR cache, skipped'); continue; }
  const before = rel(test);
  // protocol 1: fit on training forms, test on test forms
  const cTrain = fitIsotonic(train);
  const afterTrain = rel(test, cTrain);
  // protocol 2: cross-fit over test pages
  const oof: { conf: number; ok: boolean }[] = [];
  for (let k = 0; k < FOLDS; k++) {
    const c = fitIsotonic(test.filter((s) => s.page % FOLDS !== k));
    for (const s of test.filter((s) => s.page % FOLDS === k)) oof.push({ conf: applyCurve(c, s.score), ok: s.ok });
  }
  const afterCross = reliability(oof);
  const cAll = fitIsotonic(test);
  cal.curves[name] = ship === 'train' ? cTrain : cAll;
  const acc = (s: S[]) => s.filter((x) => x.ok).length / s.length;
  report.readers[name] = {
    fields: { training: train.length, testing: test.length },
    shareCorrect: { training: acc(train), testing: acc(test) },
    before,
    trainToTest: { ...afterTrain, curve: cTrain },
    crossFit: { ...afterCross, folds: FOLDS, curve: cAll },
  };
  const top = (r: ReturnType<typeof reliability>) => { const b = r.bins.filter((x) => x.fields).at(-1)!; return `top bin ${r3(b.meanConfidence)} shown / ${r3(b.correct)} right (n=${b.fields})`; };
  console.log(`\n${name}: ${train.length} training fields (${r3(acc(train))} right), ${test.length} test fields (${r3(acc(test))} right)`);
  console.log(`  before       ECE ${r3(before.ece)}  Brier ${r3(before.brier)}  ${top(before)}`);
  console.log(`  train->test  ECE ${r3(afterTrain.ece)}  Brier ${r3(afterTrain.brier)}  ${top(afterTrain)}`);
  console.log(`  cross-fit    ECE ${r3(afterCross.ece)}  Brier ${r3(afterCross.brier)}  ${top(afterCross)}`);
}
cal.fittedOn = ship === 'train' ? 'FUNSD training split (149 forms)' : 'FUNSD test split (50 forms); the reported error is cross-fitted (5 folds by page)';
cal.note = 'No separate curve for the second-reader blend: it uses the Tesseract curve. Not fitted on identity documents.';
fs.writeFileSync(path.join(ROOT, 'src/core/calibration.json'), JSON.stringify(cal, null, 1) + '\n');
const resPath = path.join(ROOT, 'public/eval/results.json');
const R = JSON.parse(fs.readFileSync(resPath, 'utf8'));
R.calibration = report;
fs.writeFileSync(resPath, JSON.stringify(R, null, 1));
console.log('\nwrote src/core/calibration.json and the calibration block of public/eval/results.json');
