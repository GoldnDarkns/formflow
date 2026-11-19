// Train the document sorter on real documents.
//
// Data (all outside the repository):
//   ../data/own-cache/train/<type>/*.json   words read from the owner's real documents (made by eval-real.ts)
//   ../data/ocr/training-full/*.json        149 FUNSD training forms          -> "form"
//   ../data/forms20/ocr-tesseract/*.json    20 filled public forms            -> "form"
//
// Model: multinomial logistic regression over the fixed clue list in
// src/core/classify.ts. The weights start from, and are pulled towards, the
// hand-set rule weights (L2 distance to the prior), so a type with no real
// examples (visa, plain ID card) keeps its hand-written rules.
//
// Honest measurement: leave-one-out over the real documents. Each real document
// is sorted by a model that never saw it. The hand-set rules are scored on the
// same documents for comparison. A separate held-out folder (../data/own/test)
// is scored by eval-real.ts once the owner supplies it.
//
// Privacy: only one number per clue and type is written to src/core/sorter.json.
// Only counts are written to public/eval/results.json.
//
// usage: npx tsx scripts/train-sorter.ts
import fs from 'node:fs';
import path from 'node:path';
import { CLUES, DOC_TYPES, classifyFeatures, priorModel, sorterFeatures, type DocType, type SorterModel } from '../src/core/classify';
import { cleanWords } from '../src/core/ocr';
import type { MrzResult } from '../src/core/mrz';
import { DATA } from './funsd';

const ROOT = path.resolve(import.meta.dirname, '..');
interface Sample { x: number[]; y: number; real: boolean; file: string; synth?: boolean }
const K = DOC_TYPES.length, D = CLUES.length;
const cls = (label: string) => DOC_TYPES.indexOf((DOC_TYPES.includes(label as DocType) ? label : 'other') as DocType);

/** the few facts about the code lines that the clues need, rebuilt from the private cache */
const mrzOf = (m: any): MrzResult | null => (m ? ({ format: m.format, docType: m.docType, valid: m.valid, lines: [], checks: Array.from({ length: Math.max(1, m.checks ?? 1) }, (_, i) => ({ name: String(i), ok: i < (m.checksOk ?? 0) })), fields: [{ key: 'Issuing country', value: m.country ?? '' }] } as MrzResult) : null);

const samples: Sample[] = [];
const realDir = path.join(DATA, 'own-cache/train');
for (const label of fs.existsSync(realDir) ? fs.readdirSync(realDir) : [])
  for (const f of fs.readdirSync(path.join(realDir, label))) {
    const j = JSON.parse(fs.readFileSync(path.join(realDir, label, f), 'utf8'));
    samples.push({ x: sorterFeatures(j.words, j.width, j.height, mrzOf(j.mrz)), y: cls(label), real: true, file: f.replace('.json', '') });
  }
for (const dir of ['ocr/training-full', 'forms20/ocr-tesseract']) {
  const d = path.join(DATA, dir);
  if (!fs.existsSync(d)) continue;
  for (const f of fs.readdirSync(d)) {
    const j = JSON.parse(fs.readFileSync(path.join(d, f), 'utf8'));
    samples.push({ x: sorterFeatures(cleanWords(j.words), j.width, j.height, null), y: cls('form'), real: false, file: f });
  }
}
// made-up specimens (scripts/make-specimens.py): the train half is learned from, the test half only scored
const synthDocs = (split: string): Sample[] => {
  const d = path.join(DATA, 'synth-cache', split), out: Sample[] = [];
  for (const label of fs.existsSync(d) ? fs.readdirSync(d) : [])
    for (const f of fs.readdirSync(path.join(d, label))) {
      const j = JSON.parse(fs.readFileSync(path.join(d, label, f), 'utf8'));
      out.push({ x: sorterFeatures(j.words, j.width, j.height, mrzOf(j.mrz)), y: cls(label), real: false, synth: true, file: `${label}/${f.replace('.json', '.jpg')}` });
    }
  return out;
};
const synthTrain = synthDocs('train'), synthTest = synthDocs('test');
samples.push(...synthTrain);
const real = samples.filter((s) => s.real);
console.log(`${samples.length} training pages: ${real.length} real documents, ${synthTrain.length} made-up specimens, ${samples.length - real.length - synthTrain.length} public forms; ${synthTest.length} specimens held back for testing`);

const prior = priorModel();
const LAMBDA = 0.02;
function fit(train: Sample[]): SorterModel {
  const W = prior.weights.map((r) => [...r]), b = [...prior.bias];
  // every class counts the same, however many pages it has
  const count = new Array(K).fill(0);
  for (const s of train) count[s.y]++;
  const present = count.filter((c) => c > 0).length;
  const wt = train.map((s) => 1 / (count[s.y] * present));
  for (let it = 0; it < 1500; it++) {
    const gW = W.map((r, k) => r.map((v, j) => 2 * LAMBDA * (v - prior.weights[k][j])));
    const gb = b.map((v, k) => 2 * LAMBDA * (v - prior.bias[k]));
    train.forEach((s, n) => {
      const z = W.map((r, k) => b[k] + r.reduce((a, v, j) => a + v * s.x[j], 0));
      const mx = Math.max(...z);
      const e = z.map((v) => Math.exp(v - mx));
      const Z = e.reduce((a, v) => a + v, 0);
      for (let k = 0; k < K; k++) {
        const g = (e[k] / Z - (k === s.y ? 1 : 0)) * wt[n];
        gb[k] += g;
        for (let j = 0; j < D; j++) if (s.x[j]) gW[k][j] += g * s.x[j];
      }
    });
    const lr = 0.8;
    for (let k = 0; k < K; k++) { b[k] -= lr * gb[k]; for (let j = 0; j < D; j++) W[k][j] -= lr * gW[k][j]; }
  }
  const r3 = (v: number) => Math.round(v * 1000) / 1000;
  return { fitted: true, clues: CLUES.map((c) => c.id), types: DOC_TYPES, weights: W.map((r) => r.map(r3)), bias: b.map(r3) };
}

// ---- leave-one-out over the real documents ----
const byType: Record<string, { n: number; rules: number; learned: number }> = {};
const wrong: string[] = [];
for (const s of real) {
  const t = DOC_TYPES[s.y];
  byType[t] ??= { n: 0, rules: 0, learned: 0 };
  byType[t].n++;
  const r = classifyFeatures(s.x, prior).type;
  const l = classifyFeatures(s.x, fit(samples.filter((o) => o !== s))).type;
  if (r === t) byType[t].rules++;
  if (l === t) byType[t].learned++; else wrong.push(`${s.file} -> ${l}`);
}
const tot = Object.values(byType).reduce((a, v) => ({ n: a.n + v.n, rules: a.rules + v.rules, learned: a.learned + v.learned }), { n: 0, rules: 0, learned: 0 });
console.log('\ntype'.padEnd(19) + 'n'.padStart(4) + 'hand rules'.padStart(12) + 'learned (left-out)'.padStart(20));
for (const [t, v] of Object.entries(byType)) console.log(t.padEnd(18) + String(v.n).padStart(4) + String(v.rules).padStart(12) + String(v.learned).padStart(20));
console.log('all'.padEnd(18) + String(tot.n).padStart(4) + `${tot.rules} (${Math.round((100 * tot.rules) / tot.n)}%)`.padStart(12) + `${tot.learned} (${Math.round((100 * tot.learned) / tot.n)}%)`.padStart(20));
console.log('\nstill wrong when left out:\n  ' + wrong.join('\n  '));

// ---- public forms must still be sorted as forms ----
const forms = samples.filter((s) => !s.real && !s.synth);
const final = fit(samples);
const formsOk = forms.filter((s) => classifyFeatures(s.x, final).type === 'form').length;
const formsRules = forms.filter((s) => classifyFeatures(s.x, prior).type === 'form').length;
console.log(`\npublic forms sorted as form: hand rules ${formsRules}/${forms.length}, learned ${formsOk}/${forms.length} (these were in training)`);

// ---- held-back specimens: never used for fitting; some are from countries the training set never had ----
let synth: any = null;
if (synthTest.length) {
  const tr = JSON.parse(fs.readFileSync(path.join(DATA, 'synth/test/truth.json'), 'utf8')).documents as any[];
  const seen = new Set((JSON.parse(fs.readFileSync(path.join(DATA, 'synth/train/truth.json'), 'utf8')).documents as any[]).map((d) => d.country));
  const info = new Map(tr.map((d) => [d.file, d]));
  const tally = (pick: (s: Sample) => boolean) => { const rs = synthTest.filter(pick); return { n: rs.length, rules: rs.filter((s) => classifyFeatures(s.x, prior).type === DOC_TYPES[s.y]).length, learned: rs.filter((s) => classifyFeatures(s.x, final).type === DOC_TYPES[s.y]).length }; };
  synth = { train: synthTrain.length, test: synthTest.length, all: tally(() => true), newCountries: tally((s) => !seen.has(info.get(s.file)?.country)), seenCountries: tally((s) => seen.has(info.get(s.file)?.country)), byType: Object.fromEntries(DOC_TYPES.map((t, k) => [t, tally((s) => s.y === k)]).filter(([, v]: any) => v.n)), byShot: Object.fromEntries(['flat', 'desk', 'glare', 'blur', 'sideways'].map((m) => [m, tally((s) => info.get(s.file)?.shot === m)])) };
  const pc = (v: { n: number; rules: number; learned: number }) => `${v.n} pages: hand rules ${v.rules} (${Math.round((100 * v.rules) / Math.max(1, v.n))}%), learned ${v.learned} (${Math.round((100 * v.learned) / Math.max(1, v.n))}%)`;
  console.log('\nheld-back specimens  ' + pc(synth.all));
  console.log('  countries never in training  ' + pc(synth.newCountries));
  for (const [t, v] of Object.entries(synth.byType)) console.log('  ' + t.padEnd(18) + pc(v as any));
  for (const [t, v] of Object.entries(synth.byShot)) console.log('  ' + ('shot: ' + t).padEnd(18) + pc(v as any));
}
final.note = `Fitted on ${real.length} real documents, ${synthTrain.length} made-up specimens and ${forms.length} public forms; weights pulled towards the hand-set rules (lambda ${LAMBDA}).`;
fs.writeFileSync(path.join(ROOT, 'src/core/sorter.json'), JSON.stringify(final) + '\n');
const resPath = path.join(ROOT, 'public/eval/results.json');
const R = JSON.parse(fs.readFileSync(resPath, 'utf8'));
// how the whole pipeline did on the same documents (counts only), from eval-real.ts
let pipeline: any = null;
try {
  const rows = JSON.parse(fs.readFileSync(path.join(DATA, 'own-results-train.json'), 'utf8')).rows as any[];
  const mrzPages = rows.filter((r) => /^pp-|^eid-.*(back|together|screenshot)/.test(r.file));
  pipeline = { sortedRightWithShippedModel: rows.filter((r) => r.pred === (DOC_TYPES.includes(r.label) ? r.label : 'other')).length, turnedUpright: rows.filter((r) => r.turned).length, codeLinePages: mrzPages.length, codeLinesFound: mrzPages.filter((r) => r.mrz !== 'none').length, codeLinesAllChecksPass: mrzPages.filter((r) => r.mrz === 'valid').length, medianMsPerPage: [...rows].sort((a, b) => a.ms - b.ms)[rows.length >> 1]?.ms };
} catch {}
// the first run of the unchanged app on these documents, before any fix (19 Nov 2026)
const firstRun = { sortedRight: 28, codeLinesAllChecksPass: 4, turnedUpright: 0 };
// the untouched specimen set (never looked at while the reading rules were improved), whole pipeline, counts only
let finalSet: any = null;
try {
  const rows = JSON.parse(fs.readFileSync(path.join(DATA, 'synth-results-final.json'), 'utf8')).rows as any[];
  const facts = rows.flatMap((r) => Object.entries(r.facts ?? {}).map(([k, v]) => ({ k, v, t: r.label })));
  const share = (xs: any[], f: (x: any) => boolean) => ({ n: xs.length, right: xs.filter(f).length });
  finalSet = {
    documents: rows.length, sortedRight: rows.filter((r) => r.pred === r.label).length,
    codeLinePages: rows.filter((r) => r.hasMrz).length, codeLinesAllChecksPass: rows.filter((r) => r.hasMrz && r.mrz === 'valid').length,
    falseCodeLines: rows.filter((r) => !r.hasMrz && r.mrz !== 'none').length,
    neededTurning: rows.filter((r) => r.shot === 'sideways').length, turnedRight: rows.filter((r) => r.shot === 'sideways' && r.turnRight).length,
    keyFacts: share(facts, (x) => x.v === 'right'), shownVerifiedButWrong: facts.filter((x) => x.v === 'verified-wrong').length,
    byFact: Object.fromEntries(['number', 'Full name', 'Date of birth', 'Expiry date'].map((k) => [k, share(facts.filter((x) => x.k === k), (x) => x.v === 'right')])),
    byType: Object.fromEntries([...new Set(facts.map((x) => x.t))].map((t) => [t, share(facts.filter((x) => x.t === t), (x) => x.v === 'right')])),
    byShot: Object.fromEntries(['flat', 'desk', 'glare', 'blur', 'sideways'].map((m) => [m, share(rows.filter((r) => r.shot === m), (r) => r.pred === r.label)])),
  };
} catch {}
R.sorter_real = { generated: new Date().toISOString(), pipeline, firstRun, synth, finalSet, note: 'Real identity documents, used with consent and kept outside the repository. Only counts are published. Learned = leave-one-out: each document is sorted by a model that never saw it.', documents: tot.n, publicForms: forms.length, byType, handRules: tot.rules / tot.n, learnedLeftOut: tot.learned / tot.n };
fs.writeFileSync(resPath, JSON.stringify(R, null, 1));
console.log('\nwrote src/core/sorter.json and the sorter_real block of public/eval/results.json');
