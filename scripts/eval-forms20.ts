// Score the app's pipeline on the extra test set: 20 public forms filled in
// with invented data (testsets/forms20, built by scripts/make-forms20.py).
// Unlike FUNSD these are modern, cleanly typed forms, half of them degraded to
// look like an office scan, so this is a test of how results transfer.
//
// For each value typed into a form we ask three things, each stricter:
//   read      the reader got the text in that box right (>= 80% similar)
//   extracted that text came out as the value of a field
//   labelled  ... and the field's label matches the label printed by the box
//
// usage: npx tsx scripts/eval-forms20.ts        (OCR is cached in ../data/forms20)
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { PNG } from 'pngjs';
import * as ortNode from 'onnxruntime-node';
import { createWorker, PSM } from 'tesseract.js';
import type { Box, OcrWord } from '../src/core/types';
import { rgbaToGray, grayToRgba, preprocess, resize, ocrScale } from '../src/core/preprocess';
import { wordsFromTesseract, cleanWords } from '../src/core/ocr';
import { loadPpOcr } from '../src/core/ppocr';
import { analyse } from '../src/core/pipeline';
import { loadClassifiers } from '../src/core/classifier';
import { classifyDocument } from '../src/core/classify';
import { reliability } from '../src/core/calibration';
import { clean, sim } from './match';
import { DATA } from './funsd';

const ROOT = path.resolve(import.meta.dirname, '..');
const SET = path.join(ROOT, 'testsets/forms20');
const truth = JSON.parse(fs.readFileSync(path.join(SET, 'truth.json'), 'utf8'));
type Reader = 'tesseract' | 'ppocr';
interface Cached { width: number; height: number; ms: number; words: OcrWord[] }

async function gray(file: string) {
  const { data, info } = await sharp(path.join(SET, file)).greyscale().raw().toBuffer({ resolveWithObject: true });
  const rgba = new Uint8Array(info.width * info.height * 4);
  for (let i = 0; i < data.length; i++) { rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = data[i]; rgba[i * 4 + 3] = 255; }
  return rgbaToGray(rgba, info.width, info.height);
}

async function ocrAll(reader: Reader): Promise<Map<string, Cached>> {
  const dir = path.join(DATA, 'forms20', 'ocr-' + reader);
  fs.mkdirSync(dir, { recursive: true });
  const out = new Map<string, Cached>();
  let worker: Awaited<ReturnType<typeof createWorker>> | null = null;
  let pp: Awaited<ReturnType<typeof loadPpOcr>> | null = null;
  for (const f of truth.forms) {
    const cache = path.join(dir, f.file.replace(/\.\w+$/, '.json'));
    if (fs.existsSync(cache)) { const j = JSON.parse(fs.readFileSync(cache, 'utf8')); out.set(f.file, { ...j, words: cleanWords(j.words) }); continue; }
    let g = await gray(f.file);
    const W = g.width, H = g.height;
    const scale = ocrScale(W, H);
    let words: OcrWord[];
    const t = Date.now();
    if (reader === 'tesseract') {
      // same steps as the app: resize, deskew, binarise, remove ruling lines, sparse-text Tesseract
      ({ img: g } = preprocess(resize(g, scale)));
      worker ??= await createWorker('eng', 1, { langPath: path.join(ROOT, 'node_modules/@tesseract.js-data/eng/4.0.0_best_int'), cacheMethod: 'none' });
      await worker.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: '1' });
      const enc = new PNG({ width: g.width, height: g.height });
      enc.data = Buffer.from(grayToRgba(g));
      const { data } = await worker.recognize(PNG.sync.write(enc), {}, { blocks: true });
      words = wordsFromTesseract(data.blocks as any, scale);
    } else {
      ({ img: g } = preprocess(resize(g, scale), { deskew: true, binarise: false, removeLines: false }));
      pp ??= await loadPpOcr(ortNode as any, path.join(ROOT, 'public/models/ppocr/det.onnx'), path.join(ROOT, 'public/models/ppocr/rec.onnx'), fs.readFileSync(path.join(ROOT, 'public/models/ppocr/dict.txt'), 'utf8'), undefined, { maxSide: 1600, boxThresh: 0.55, unclip: 0.8 });
      words = (await pp.read(grayToRgba(g), g.width, g.height)).map((w) => ({ ...w, box: w.box.map((v) => v / scale) as Box }));
    }
    const c = { file: f.file, width: W, height: H, ms: Date.now() - t, words };
    fs.writeFileSync(cache, JSON.stringify(c));
    out.set(f.file, { ...c, words: cleanWords(words) });
    process.stdout.write(`${reader} ${f.file} ${words.length} words ${c.ms} ms\n`);
  }
  await worker?.terminate();
  return out;
}

const clf = await loadClassifiers(ortNode as any, path.join(ROOT, 'public/models/entity_gbdt.onnx'), path.join(ROOT, 'public/models/link_gbdt.onnx'));
const inter = (a: Box, b: Box) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
const area = (b: Box) => Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
// the label printed on the form is often longer than what the app keeps as a key
const keyOk = (pred: string, label: string) => {
  const p = clean(pred), l = clean(label);
  if (!p || !l) return false;
  return sim(p, l) >= 0.6 || (p.length >= 5 && (l.includes(p) || p.includes(l)));
};

interface Tot { forms: number; asForm: number; values: number; read: number; extracted: number; labelled: number; fields: number; ms: number }
const zero = (): Tot => ({ forms: 0, asForm: 0, values: 0, read: 0, extracted: 0, labelled: 0, fields: 0, ms: 0 });
const rate = (t: Tot) => ({ forms: t.forms, sortedAsForm: t.asForm, typedValues: t.values, read: t.read / t.values, extracted: t.extracted / t.values, labelled: t.labelled / t.values, fieldsFound: t.fields, msPerPage: t.ms / t.forms });

const R: any = { generated: new Date().toISOString(), set: '20 public US federal forms (IRS, OPM) filled in with invented data; odd-numbered forms degraded to look like a scan', readers: {} };
for (const reader of ['tesseract', 'ppocr'] as Reader[]) {
  const ocr = await ocrAll(reader);
  const groups: Record<string, Tot> = { all: zero(), clean: zero(), 'scan-like': zero() };
  const conf: { conf: number; ok: boolean }[] = [];
  const perForm: any[] = [];
  const misses: any[] = [];
  for (const f of truth.forms) {
    const o = ocr.get(f.file)!;
    const an = await analyse(o.words, o.width, o.height, clf, undefined, 'form', reader);
    const type = classifyDocument(o.words, o.width, o.height, null).type;
    const t = zero();
    t.forms = 1; t.asForm = type === 'form' ? 1 : 0; t.ms = o.ms; t.fields = an.fields.filter((x) => x.value).length;
    const hitField = new Map<number, boolean>(); // field index -> right (value and label)
    for (const g of f.fields as { label: string; value: string; box: Box }[]) {
      t.values++;
      const inBox = o.words.filter((w) => inter(w.box, g.box) / area(w.box) > 0.5).sort((a, b) => a.box[0] - b.box[0]).map((w) => w.text).join(' ');
      const readOk = sim(inBox, g.value) >= 0.8;
      if (readOk) t.read++;
      let ext = false, lab = false;
      an.fields.forEach((fl, i) => {
        if (!fl.value) return;
        const boxes = an.links.filter((l) => l.q === fl.qId).map((l) => an.entities[l.a].box);
        if (!boxes.some((b) => inter(b, g.box) / Math.min(area(b), area(g.box)) > 0.3)) return;
        const vOk = sim(fl.value, g.value) >= 0.8;
        const kOk = vOk && keyOk(fl.key, g.label);
        if (vOk) ext = true;
        if (kOk) lab = true;
        hitField.set(i, (hitField.get(i) ?? false) || kOk);
      });
      if (ext) t.extracted++;
      if (lab) t.labelled++;
      if (readOk && !ext && misses.length < 12) misses.push({ file: f.file, label: g.label.slice(0, 60), value: g.value, why: 'read correctly but not linked to a label' });
      else if (!readOk && misses.length < 24 && misses.filter((m) => m.why.startsWith('mis')).length < 12) misses.push({ file: f.file, label: g.label.slice(0, 60), value: g.value, why: `misread as "${inBox.slice(0, 40)}"` });
    }
    // calibration transfer: fields that sit on a typed value, shown confidence vs right
    for (const [i, ok] of hitField) conf.push({ conf: an.fields[i].confidence, ok });
    for (const k of ['all', f.quality]) for (const key of Object.keys(t) as (keyof Tot)[]) groups[k][key] += t[key];
    perForm.push({ file: f.file, quality: f.quality, type, values: t.values, read: t.read, extracted: t.extracted, labelled: t.labelled });
  }
  const rel = reliability(conf, 5);
  R.readers[reader] = {
    all: rate(groups.all), clean: rate(groups.clean), scanLike: rate(groups['scan-like']),
    confidence: { fields: conf.length, meanShown: conf.reduce((a, x) => a + x.conf, 0) / Math.max(1, conf.length), shareRight: conf.filter((x) => x.ok).length / Math.max(1, conf.length), ece: rel.ece, bins: rel.bins },
    perForm, examples: misses,
  };
  const pc = (v: number) => (v * 100).toFixed(1) + '%';
  for (const [k, g] of Object.entries(groups)) console.log(`${reader.padEnd(9)} ${k.padEnd(9)} forms ${g.forms} as-form ${g.asForm}  values ${g.values}  read ${pc(g.read / g.values)}  extracted ${pc(g.extracted / g.values)}  labelled ${pc(g.labelled / g.values)}  ${Math.round(g.ms / g.forms)} ms/page`);
  console.log(`${reader.padEnd(9)} confidence on ${conf.length} fields: shown ${pc(R.readers[reader].confidence.meanShown)}, right ${pc(R.readers[reader].confidence.shareRight)}, ECE ${rel.ece.toFixed(3)}`);
}
const resPath = path.join(ROOT, 'public/eval/results.json');
const all = JSON.parse(fs.readFileSync(resPath, 'utf8'));
all.forms20 = R;
fs.writeFileSync(resPath, JSON.stringify(all, null, 1));
console.log('wrote the forms20 block of public/eval/results.json');
