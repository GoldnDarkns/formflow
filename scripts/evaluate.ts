// Full evaluation on the FUNSD test split (50 forms) + optional own test set.
// usage: npx tsx scripts/evaluate.ts
import fs from 'node:fs';
import path from 'node:path';
import * as ortNode from 'onnxruntime-node';
import type { Entity, Field, SecondReading } from '../src/core/types';
import { analyse, wordAgreement, type Classifiers } from '../src/core/pipeline';
import { loadClassifiers } from '../src/core/classifier';
import { argmaxLabel, candidatePairs, decodeLinks, entityFeatures, pageCtx, pairFeatures } from '../src/core/layout';
import { scoreOcr } from '../src/core/ocr';
import { DATA, gtAsEntities, loadGt, loadOcr, pages } from './funsd';
import { gtFields, matchFields, clean, sim } from './match';

const ROOT = path.resolve(import.meta.dirname, '..');
const ort = ortNode as any;
const clf = await loadClassifiers(ort, path.join(ROOT, 'public/models/entity_gbdt.onnx'), path.join(ROOT, 'public/models/link_gbdt.onnx'));
const clfLR = await loadClassifiers(ort, path.join(DATA, 'features/entity_logreg.onnx'), path.join(DATA, 'features/link_logreg.onnx'));

// Rule baseline (geometry + colon heuristics), same interface as the models.
const baseline: Classifiers = {
  async entity(X) {
    return X.map((f) => {
      const endsColon = f[8], leftColon = f[27], aboveColon = f[32], allCaps = f[6], nWords = Math.expm1(f[0]), sameLine = f[33];
      if (endsColon) return [1, 0, 0, 0];
      if (leftColon || aboveColon) return [0, 1, 0, 0];
      if (allCaps && nWords <= 6 && sameLine === 0) return [0, 0, 1, 0];
      return [0, 0, 0, 1];
    });
  },
  async link(X) {
    return X.map((f) => (f[13] === 0 && ((f[9] && f[10]) || f[11]) ? 0.9 : 0.1));
  },
};

const files = pages('testing');
const R: any = { dataset: 'FUNSD test (50 forms)', generated: new Date().toISOString() };

// ---------- 1. OCR quality ----------
function ocrBlock(mode: string) {
  let tot = { gtChars: 0, errors: 0, gtWords: 0, omitted: 0, predWords: 0, hallucinated: 0, correct: 0, ms: 0, n: 0 };
  for (const f of files) {
    const o = loadOcr('testing', mode, f);
    if (!o) continue;
    const gt = loadGt('testing', f);
    const s = scoreOcr(gt.ents.flatMap((e) => e.words), o.words);
    tot.gtChars += s.gtChars; tot.errors += s.errors; tot.gtWords += s.gtWords; tot.omitted += s.omitted;
    tot.predWords += s.predWords; tot.hallucinated += s.hallucinated; tot.correct += s.wordAcc * s.gtWords; tot.ms += o.ms; tot.n++;
  }
  return summarise(tot);
}
function summarise(t: any) {
  return {
    pages: t.n,
    cer: t.errors / t.gtChars,
    wordAccuracy: t.correct / t.gtWords,
    omissionRate: t.omitted / t.gtWords,
    hallucinationRate: t.hallucinated / Math.max(1, t.predWords),
    msPerPage: t.ms / Math.max(1, t.n),
  };
}
R.ocr = {} as any;
for (const m of ['raw', 'upscale', 'full']) R.ocr['tesseract_' + m] = ocrBlock(m);
R.ocr.ppocr_v5_mobile = ocrBlock('pp');

function loadFlor(tiles: string, f: string): { ms: number; regions: SecondReading[] } | null {
  const p = path.join(DATA, 'florence', `testing-${tiles}`, f);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null;
}
// Florence reads lines, not words: split each region's text over its box proportionally.
function florenceWords(rs: SecondReading[]) {
  const out: { text: string; box: [number, number, number, number] }[] = [];
  for (const r of rs) {
    const ws = r.text.split(/\s+/).filter(Boolean);
    const total = ws.reduce((a, w) => a + w.length + 1, 0);
    let x = r.box[0];
    const W = r.box[2] - r.box[0];
    for (const w of ws) {
      const dw = (W * (w.length + 1)) / total;
      out.push({ text: w, box: [x, r.box[1], x + dw, r.box[3]] });
      x += dw;
    }
  }
  return out;
}
for (const tiles of ['1x1', '1x2']) {
  let tot = { gtChars: 0, errors: 0, gtWords: 0, omitted: 0, predWords: 0, hallucinated: 0, correct: 0, ms: 0, n: 0 };
  for (const f of files) {
    const fl = loadFlor(tiles, f);
    if (!fl) continue;
    const gt = loadGt('testing', f);
    const s = scoreOcr(gt.ents.flatMap((e) => e.words), florenceWords(fl.regions));
    tot.gtChars += s.gtChars; tot.errors += s.errors; tot.gtWords += s.gtWords; tot.omitted += s.omitted;
    tot.predWords += s.predWords; tot.hallucinated += s.hallucinated; tot.correct += s.wordAcc * s.gtWords; tot.ms += fl.ms; tot.n++;
  }
  if (tot.n) R.ocr['florence2_' + tiles] = summarise(tot);
}

// ---------- 2. Linking on ground-truth entities (standard FUNSD protocol) ----------
async function linkingGt(c: Classifiers, goldLabels: boolean) {
  let tp = 0, fp = 0, fn = 0;
  let labOk = 0, labN = 0;
  for (const f of files) {
    const o = loadOcr('testing', 'full', f)!;
    const gt = loadGt('testing', f, o);
    const ents = gtAsEntities(gt);
    const ctx = pageCtx(ents, o.width, o.height);
    let labels = ents.map((e) => e.label!);
    if (!goldLabels) {
      const pr = await c.entity(entityFeatures(ents, ctx));
      const pl = pr.map(argmaxLabel);
      pl.forEach((l, i) => { labN++; if (l === labels[i]) labOk++; });
      labels = pl;
    }
    const qi = ents.map((_, i) => i).filter((i) => labels[i] === 'question');
    const ai = ents.map((_, i) => i).filter((i) => labels[i] === 'answer');
    const pairs = candidatePairs(ents, qi, ai, ctx);
    const probs = pairs.length ? await c.link(pairFeatures(ents, pairs, ctx)) : [];
    const pred = new Set(decodeLinks(pairs, probs).map((l) => `${ents[l.q].id},${ents[l.a].id}`));
    for (const k of pred) gt.qa.has(k) ? tp++ : fp++;
    for (const k of gt.qa) if (!pred.has(k)) fn++;
  }
  const p = tp / Math.max(1, tp + fp), r = tp / Math.max(1, tp + fn);
  return { precision: p, recall: r, f1: (2 * p * r) / Math.max(1e-9, p + r), ...(goldLabels ? {} : { labelAccuracy: labOk / labN }) };
}
R.linking_gt_entities = {
  baseline_rules: { gold_labels: await linkingGt(baseline, true), predicted_labels: await linkingGt(baseline, false) },
  logistic_regression: { gold_labels: await linkingGt(clfLR, true), predicted_labels: await linkingGt(clfLR, false) },
  gradient_boosting: { gold_labels: await linkingGt(clf, true), predicted_labels: await linkingGt(clf, false) },
};

// ---------- 3. End-to-end key-value extraction from the image ----------
async function endToEnd(c: Classifiers, useFlorence?: string, mode = 'full') {
  let tp = 0, nPred = 0, nGold = 0, ms = 0, n = 0;
  const conf: { conf: number; ok: boolean; status: string }[] = [];
  let wordAgree = { agreeOk: 0, agreeN: 0, disOk: 0, disN: 0, noneOk: 0, noneN: 0 };
  const gallery: any[] = [];
  for (const f of files) {
    const o = loadOcr('testing', mode, f)!;
    const gt = loadGt('testing', f, o);
    let second: SecondReading[] | undefined;
    if (useFlorence) {
      const fl = loadFlor(useFlorence, f);
      if (!fl) continue;
      second = fl.regions;
    }
    const t = performance.now();
    const an = await analyse(o.words, o.width, o.height, c, second, 'form', mode === 'pp' ? 'ppocr' : 'tesseract');
    ms += performance.now() - t;
    n++;
    const gold = gtFields(gt);
    const m = matchFields(an.fields, gold);
    tp += m.tp; nPred += m.nPred; nGold += m.nGold;
    for (const r of m.res) conf.push({ conf: r.f.raw ?? r.f.confidence, ok: r.ok, status: r.f.status });
    if (second && an.agreement) {
      // word-level: is the Tesseract word exactly right (vs the GT word it lies in)?
      const gw = gt.ents.flatMap((e) => e.words);
      o.words.forEach((w, i) => {
        const g = gw.find((x) => {
          const ix = Math.max(0, Math.min(x.box[2], w.box[2]) - Math.max(x.box[0], w.box[0]));
          const iy = Math.max(0, Math.min(x.box[3], w.box[3]) - Math.max(x.box[1], w.box[1]));
          return ix * iy > 0.5 * (w.box[2] - w.box[0]) * (w.box[3] - w.box[1]);
        });
        if (!g) return;
        const ok = clean(g.text) === clean(w.text);
        const a = an.agreement![i];
        if (a === true) { wordAgree.agreeN++; if (ok) wordAgree.agreeOk++; }
        else if (a === false) { wordAgree.disN++; if (ok) wordAgree.disOk++; }
        else { wordAgree.noneN++; if (ok) wordAgree.noneOk++; }
      });
    }
    if (!useFlorence && c === clf && mode === 'full' && gallery.length < 40)
      for (const r of m.res.filter((r) => !r.ok).slice(0, 2)) {
        const g = gold.map((g) => ({ g, s: sim(r.f.key, g.key) })).sort((a, b) => b.s - a.s)[0];
        gallery.push({ file: f, predKey: r.f.key, predValue: r.f.value, goldKey: g?.g.key, goldValue: g?.g.value, confidence: +r.f.confidence.toFixed(2) });
      }
  }
  const p = tp / Math.max(1, nPred), r = tp / Math.max(1, nGold);
  const ok = conf.filter((x) => x.status === 'ok'), rev = conf.filter((x) => x.status !== 'ok');
  const out: any = {
    pages: n, precision: p, recall: r, f1: (2 * p * r) / Math.max(1e-9, p + r), msPerPage: ms / Math.max(1, n),
    flagged: { autoAccepted: ok.length, autoAcceptedCorrect: ok.filter((x) => x.ok).length / Math.max(1, ok.length), flaggedForReview: rev.length, flaggedCorrect: rev.filter((x) => x.ok).length / Math.max(1, rev.length) },
  };
  if (useFlorence) out.wordAgreement = {
    agree: { words: wordAgree.agreeN, accuracy: wordAgree.agreeOk / Math.max(1, wordAgree.agreeN) },
    disagree: { words: wordAgree.disN, accuracy: wordAgree.disOk / Math.max(1, wordAgree.disN) },
    secondReaderSawNothing: { words: wordAgree.noneN, accuracy: wordAgree.noneOk / Math.max(1, wordAgree.noneN) },
  };
  // reliability of the confidence index: is a "90%" field right 90% of the time?
  const bins = [[0, 0.5], [0.5, 0.65], [0.65, 0.8], [0.8, 0.9], [0.9, 1.01]].map(([lo, hi]) => {
    const b = conf.filter((x) => x.conf >= lo && x.conf < hi);
    return { range: `${Math.round(lo * 100)}–${Math.min(100, Math.round(hi * 100))}%`, fields: b.length, meanConfidence: b.reduce((a, x) => a + x.conf, 0) / Math.max(1, b.length), correct: b.filter((x) => x.ok).length / Math.max(1, b.length) };
  });
  out.reliability = bins;
  if (gallery.length) out.errorExamples = gallery;
  return out;
}
R.end_to_end = {
  baseline_rules: await endToEnd(baseline),
  logistic_regression: await endToEnd(clfLR),
  gradient_boosting: await endToEnd(clf),
  gradient_boosting_ppocr_reader: await endToEnd(clf, undefined, 'pp'),
};
for (const tiles of ['1x1', '1x2']) {
  if (fs.existsSync(path.join(DATA, 'florence', `testing-${tiles}`))) {
    const e = await endToEnd(clf, tiles);
    if (e.pages) R.end_to_end['gradient_boosting_with_florence_' + tiles] = e;
    // same pages without Florence, for a fair comparison of the review flags
    if (e.pages && e.pages < files.length) R.end_to_end['note_' + tiles] = `Florence ${tiles} ran on ${e.pages} pages`;
  }
}

// ---------- 4. Document-type recognition on the 50 forms (all should be "form") ----------
{
  const { classifyDocument } = await import('../src/core/classify');
  const counts: Record<string, number> = {};
  let confSum = 0;
  for (const f of files) {
    const o = loadOcr('testing', 'full', f)!;
    const c = classifyDocument(o.words, o.width, o.height, null);
    counts[c.type] = (counts[c.type] ?? 0) + 1;
    confSum += c.confidence;
  }
  let specimens: any;
  try { const j = JSON.parse(fs.readFileSync(path.join(DATA, 'specimens.json'), 'utf8')); specimens = { correct: j.correct, total: j.total, note: 'Synthetic specimen scans run through the built app in Chrome (scripts/e2e.mjs)' }; } catch {}
  R.doc_type = { specimens, funsd_forms: { pages: files.length, predicted: counts, accuracy: (counts.form ?? 0) / files.length, meanConfidence: confSum / files.length } };
}

for (const [k, f] of [['magic_scan', 'magic.json'], ['critic_vision', 'critic.json']]) try { R[k] = JSON.parse(fs.readFileSync(path.join(DATA, f), 'utf8')); } catch {}
try { R.training = JSON.parse(fs.readFileSync(path.join(DATA, 'features/train_report.json'), 'utf8')); } catch {}
try { R.calibration = JSON.parse(fs.readFileSync(path.join(ROOT, 'public/eval/results.json'), 'utf8')).calibration; } catch {}
fs.mkdirSync(path.join(ROOT, 'public/eval'), { recursive: true });
fs.writeFileSync(path.join(ROOT, 'public/eval/results.json'), JSON.stringify(R, null, 1));
const r3 = (x: number) => Math.round(x * 1000) / 1000;
console.log(JSON.stringify(R, (k, v) => (typeof v === 'number' ? r3(v) : k === 'errorExamples' || k === 'training' || k === 'ocr' || k === 'linking_gt_entities' ? undefined : v), 1));
