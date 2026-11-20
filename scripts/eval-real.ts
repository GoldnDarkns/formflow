// Private test on real documents kept outside the repository.
//
// Layout: ../data/own/<split>/<type>/<any name>.(jpg|png|webp)   (split = train | test)
// The folder name is the true document type. Runs the same steps as the app
// (find the page, flatten, straighten, clean, read, read the code lines, sort)
// and reports ONLY counts and neutral file names. No text read from a document
// is ever printed or written to the repository.
//
// The words read from each page are cached in ../data/own-cache (private, next to
// the documents) so the sorter can be trained without reading every page again.
//
// usage: npx tsx scripts/eval-real.ts [train|test]
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { PNG } from 'pngjs';
import { createWorker, PSM } from 'tesseract.js';
import type { GrayImage, OcrWord } from '../src/core/types';
import { rgbaToGray, grayToRgba, preprocess, resize, ocrScale, backgroundUnevenness } from '../src/core/preprocess';
import { detectPage, flattenLighting, warpQuad } from '../src/core/scan';
import { wordsFromTesseract, cleanWords } from '../src/core/ocr';
import { findMrzLines, parseMrz, type MrzResult } from '../src/core/mrz';
import { classifyDocument, DOC_TYPES, ID_TYPES } from '../src/core/classify';
import * as ortNode from 'onnxruntime-node';
import { analyse } from '../src/core/pipeline';
import { loadClassifiers } from '../src/core/classifier';
import { extractFacts } from '../src/core/facts';
import { loadPpOcr } from '../src/core/ppocr';
import { believableMrz, linesFromWords, mrzCrops, mrzNeuralCrops, readingQuality, TURNS, TURN_GAIN, UPRIGHT_MIN } from '../src/core/orient';
import { DATA } from './funsd';

const split = process.argv[2] ?? 'train';
const root = process.argv[3] ?? 'own'; // 'own' = the owner's real documents, 'synth' = made-up specimens from make-specimens.py
const ROOT = path.resolve(import.meta.dirname, '..');
const dir = path.join(DATA, root, split);
const cacheDir = path.join(DATA, `${root}-cache`, split);
// made-up specimens come with the truth, so more can be checked
const truthFile = path.join(dir, 'truth.json');
const truth: Record<string, any> = fs.existsSync(truthFile) ? Object.fromEntries(JSON.parse(fs.readFileSync(truthFile, 'utf8')).documents.map((d: any) => [d.file, d])) : {};
const langPath = path.join(ROOT, 'node_modules/@tesseract.js-data/eng/4.0.0_best_int');
const MAX_SIDE = 2400;

const tess = await createWorker('eng', 1, { langPath, cacheMethod: 'none' });
await tess.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: '1' });
const clf = await loadClassifiers(ortNode as any, path.join(ROOT, 'public/models/entity_gbdt.onnx'), path.join(ROOT, 'public/models/link_gbdt.onnx'));
const pp = await loadPpOcr(ortNode as any, path.join(ROOT, 'public/models/ppocr/det.onnx'), path.join(ROOT, 'public/models/ppocr/rec.onnx'), fs.readFileSync(path.join(ROOT, 'public/models/ppocr/dict.txt'), 'utf8'), process.env.EP ? { executionProviders: [process.env.EP] } : undefined, { maxSide: 1600, boxThresh: 0.55, unclip: 0.8 });
const CLASSICAL_ONLY = process.env.MRZ === 'classical'; // to compare against the old way
const tessMrz = await createWorker('eng', 1, { langPath, cacheMethod: 'none' });
await tessMrz.setParameters({ tessedit_pageseg_mode: PSM.SINGLE_BLOCK, tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<' });

const png = (g: GrayImage, y0 = 0, y1 = g.height) => {
  const h = Math.max(1, Math.round(y1) - Math.round(y0));
  const enc = new PNG({ width: g.width, height: h });
  const src = grayToRgba(g);
  enc.data = Buffer.from(src.buffer, Math.round(y0) * g.width * 4, h * g.width * 4);
  return PNG.sync.write(enc);
};
const mrzScore = (m: MrzResult | null) => (m ? m.checks.filter((c) => c.ok).length + (m.valid ? 10 : 0) : -1);

/** same rule as the app: rows dense with '<' mark the code lines */
function mrzBand(words: OcrWord[], scale: number, H: number): [number, number] | null {
  const lines = new Map<number, OcrWord[]>();
  for (const w of words) { if (!lines.has(w.line)) lines.set(w.line, []); lines.get(w.line)!.push(w); }
  const ys: number[] = [];
  for (const ws of lines.values()) {
    const t = ws.map((w) => w.text).join('');
    if ((t.match(/</g)?.length ?? 0) >= 3 || /[A-Z0-9]{2,}<{2,}/.test(t)) ys.push(...ws.flatMap((w) => [w.box[1], w.box[3]]));
  }
  if (!ys.length) return null;
  const y0 = Math.min(...ys) * scale, y1 = Math.max(...ys) * scale;
  const pad = (y1 - y0) * 0.6 + 10;
  return [Math.max(0, y0 - pad), Math.min(H, y1 + pad)];
}

interface Row { file: string; label: string; pred: string; conf: number; words: number; pageFound: boolean; mrz: 'none' | 'found' | 'valid'; mrzFormat?: string; ms: number; turned: number; shot?: string; hasMrz?: boolean; mrzExact?: boolean; turnRight?: boolean; facts?: Record<string, 'right' | 'wrong' | 'missing' | 'verified-wrong'> }
const rows: Row[] = [];
// Speed. Reading is the slow part, not training, so:
//   SHARD=i/n  read only every n-th page (run n copies side by side, one per processor core)
//   REUSE=1    do not read a page whose result is already in the private cache; just score it
//   EP=dml     run the neural reader on the graphics card (DirectML)
// scripts/eval-fast.mjs starts the copies and then one REUSE pass that prints the report.
const [SHARD_I, SHARD_N] = (process.env.SHARD ?? '0/1').split('/').map(Number);
const REUSE = process.env.REUSE === '1';
let pageIndex = 0;
const types = fs.existsSync(dir) ? fs.readdirSync(dir, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name) : [];
for (const label of types) {
  for (const f of fs.readdirSync(path.join(dir, label)).filter((x) => /\.(png|jpe?g|webp)$/i.test(x)).sort()) {
    if (pageIndex++ % SHARD_N !== SHARD_I) continue;
    const t0 = Date.now();
    const file = path.join(dir, label, f);
    const cacheFile = path.join(cacheDir, label, f.replace(/\.[^.]+$/, '.json'));
    const read = async () => {
    // one full reading of the page turned by `deg`, exactly as the app does it
    const readAt = async (deg: number) => {
      const base = sharp(file).rotate().flatten({ background: '#fff' });
      const turnedImg = deg ? sharp(await base.toBuffer()).rotate(deg) : base;
      const meta = await sharp(await turnedImg.clone().png().toBuffer()).metadata();
      const k = Math.min(1, MAX_SIDE / Math.max(meta.width!, meta.height!));
      const { data, info } = await sharp(await turnedImg.png().toBuffer()).resize(Math.round(meta.width! * k), Math.round(meta.height! * k)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      let color = new Uint8ClampedArray(data.buffer, data.byteOffset, data.length);
      let W = info.width, H = info.height;
      const g0 = rgbaToGray(color, W, H);
      const quad = detectPage(g0);
      if (quad) { const w = warpQuad(color, W, H, quad); color = w.data as Uint8ClampedArray<ArrayBuffer>; W = w.width; H = w.height; }
      if (quad || backgroundUnevenness(quad ? rgbaToGray(color, W, H) : g0) > 12) flattenLighting(color, W, H);
      const scale = ocrScale(W, H);
      const gray = preprocess(resize(rgbaToGray(color, W, H), scale), { deskew: true, binarise: false, removeLines: false }).img;
      const clean = preprocess(gray, { deskew: false, binarise: true, removeLines: true }).img;
      const { data: d } = await tess.recognize(png(clean), {}, { blocks: true });
      return { W, H, quad, scale, gray, clean, words: cleanWords(wordsFromTesseract(d.blocks as any, scale)) };
    };
    let page = await readAt(0), turned = 0;
    if (readingQuality(page.words) <= UPRIGHT_MIN) {
      for (const deg of TURNS) {
        const alt = await readAt(deg);
        if (readingQuality(alt.words) >= readingQuality(page.words) + TURN_GAIN) { page = alt; turned = deg; if (readingQuality(alt.words) >= 8) break; }
      }
    }
    const { W, H, quad, scale, gray, clean, words } = page;
    // code lines: several crops, keep the one whose check digits pass
    let mrz: MrzResult | null = null;
    const band = mrzBand(words, scale, gray.height);
    // the neural reader first, as in the app
    if (!CLASSICAL_ONLY && mrzCrops(band, gray.height, words.length).length) for (const [y0, y1] of mrzNeuralCrops(band, gray.height, words.length)) {
      const part: GrayImage = { width: gray.width, height: y1 - y0, data: gray.data.slice(y0 * gray.width, y1 * gray.width) };
      const lines = findMrzLines(linesFromWords(await pp.read(grayToRgba(part), part.width, part.height)));
      const m = lines.length >= 2 ? parseMrz(lines) : null;
      if (believableMrz(m, !!band) && mrzScore(m) > mrzScore(mrz)) mrz = m;
      if (mrz?.valid) break;
    }
    if (!mrz?.valid) for (const cr of mrzCrops(band, gray.height, words.length)) {
      const r = await tessMrz.recognize(png(cr.binarised ? clean : gray, cr.y0, cr.y1));
      const lines = findMrzLines(r.data.text);
      const m = lines.length >= 2 ? parseMrz(lines) : null;
      if (believableMrz(m, !!band) && mrzScore(m) > mrzScore(mrz)) mrz = m;
      if (mrz?.valid) break;
    }
      // second reading of the whole page with the neural reader, for comparing the two (as the app does for identity documents)
      const second = cleanWords((await pp.read(grayToRgba(gray), gray.width, gray.height)).map((w) => ({ ...w, box: w.box.map((v) => v / scale) as OcrWord['box'] })));
      return { W, H, words, second, mrz, pageFound: !!quad, turned, ms: Date.now() - t0 };
    };
    let got: Awaited<ReturnType<typeof read>>;
    if (REUSE && fs.existsSync(cacheFile)) {
      const j = JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
      got = { W: j.width, H: j.height, words: j.words, second: j.second ?? [], mrz: j.mrzFull ?? null, pageFound: !!j.meta?.pageFound, turned: j.meta?.turned ?? 0, ms: j.meta?.ms ?? 0 };
    } else {
      got = await read();
      fs.mkdirSync(path.join(cacheDir, label), { recursive: true });
      const m = got.mrz;
      fs.writeFileSync(cacheFile, JSON.stringify({ width: got.W, height: got.H, words: got.words, second: got.second, meta: { pageFound: got.pageFound, turned: got.turned, ms: got.ms }, mrzFull: m, mrz: m ? { format: m.format, docType: m.docType, valid: m.valid, checks: m.checks.length, checksOk: m.checks.filter((c) => c.ok).length, country: m.fields.find((x) => x.key === 'Issuing country')?.value } : null }));
    }
    const { W, H, words, mrz, turned } = got;
    const cls = classifyDocument(words, W, H, mrz);
    // key facts against the truth (specimens only): the number, the dates and the name
    let factHits: Row['facts'];
    const tr = truth[`${label}/${f}`];
    if (tr) {
      const an = await analyse(words, W, H, clf, undefined, ID_TYPES.includes(cls.type) ? 'id' : 'form', 'tesseract');
      const facts = extractFacts(cls.type, words, an.fields, mrz, got.second ?? []);
      const nrm = (v: string) => v.toUpperCase().replace(/[^A-Z0-9]/g, '');
      factHits = {};
      for (const [k, v] of Object.entries(tr.fields as Record<string, string>)) {
        const group = /number/i.test(k) ? 'number' : k;
        // the same fact by name; a passport or card number may also come from the code lines under either name
        const got = facts.find((x) => x.key === k) ?? (/^(Passport|Document) number$/.test(k) ? facts.find((x) => /^(Passport|Document) number$/.test(x.key)) : undefined);
        factHits[group] = !got ? 'missing' : nrm(got.value) === nrm(v) ? 'right' : got.valid ? 'verified-wrong' : 'wrong';
      }
    }
    rows.push({ file: f.replace(/\.[^.]+$/, ''), label, pred: cls.type, conf: cls.confidence, words: words.length, pageFound: got.pageFound, mrz: mrz ? (mrz.valid ? 'valid' : 'found') : 'none', mrzFormat: mrz?.format, ms: got.ms, turned, facts: factHits, ...(truth[`${label}/${f}`] ? { shot: truth[`${label}/${f}`].shot, hasMrz: !!truth[`${label}/${f}`].mrz, mrzExact: !!truth[`${label}/${f}`].mrz && !!mrz && mrz.lines.join('|') === truth[`${label}/${f}`].mrz.join('|'), turnRight: turned === truth[`${label}/${f}`].turned } : {}) });
    process.stdout.write('.');
  }
}
await tess.terminate();
await tessMrz.terminate();
if (SHARD_N > 1) process.exit(0);

// the app has no "vehicle licence" type yet: for it, "other" is the best it can say today
const want = (label: string) => (DOC_TYPES.includes(label as any) ? label : 'other');
const ok = (r: Row) => r.pred === want(r.label);
console.log(`\n\n${split}: ${rows.length} documents, sorted right ${rows.filter(ok).length} (${((100 * rows.filter(ok).length) / Math.max(1, rows.length)).toFixed(0)}%)\n`);
console.log('true type'.padEnd(18) + 'n'.padStart(4) + 'right'.padStart(7) + '   sorted as');
for (const label of types) {
  const rs = rows.filter((r) => r.label === label);
  if (!rs.length) continue;
  const as: Record<string, number> = {};
  for (const r of rs) as[r.pred] = (as[r.pred] ?? 0) + 1;
  console.log(label.padEnd(18) + String(rs.length).padStart(4) + String(rs.filter(ok).length).padStart(7) + '   ' + Object.entries(as).sort((a, b) => b[1] - a[1]).map(([t, n]) => `${t} ${n}`).join(', '));
}
const hasTruth = Object.keys(truth).length > 0;
const withMrz = hasTruth ? rows.filter((r) => r.hasMrz) : rows.filter((r) => /pp-|eid-.*back|eid-.*together|eid-.*screenshot/.test(r.file));
console.log(`
code lines (${withMrz.length} pages that have them): found ${withMrz.filter((r) => r.mrz !== 'none').length}, every check digit passes ${withMrz.filter((r) => r.mrz === 'valid').length}` + (hasTruth ? `, read exactly right ${withMrz.filter((r) => r.mrzExact).length}; false code lines on pages without any: ${rows.filter((r) => !r.hasMrz && r.mrz !== 'none').length}` : ''));
console.log(`turned upright: ${rows.filter((r) => r.turned).length}` + (hasTruth ? ` (pages that needed turning: ${rows.filter((r) => r.shot === 'sideways').length}, turned the right way: ${rows.filter((r) => r.shot === 'sideways' && r.turnRight).length}; turned by mistake: ${rows.filter((r) => r.shot !== 'sideways' && r.turned).length})` : ''));
console.log(`page outline found in ${rows.filter((r) => r.pageFound).length} of ${rows.length}; median ${[...rows].sort((a, b) => a.ms - b.ms)[rows.length >> 1]?.ms} ms per page
`);
if (hasTruth) {
  console.log('how it was photographed'.padEnd(26) + 'n'.padStart(4) + 'sorted right'.padStart(14));
  for (const m of ['flat', 'desk', 'glare', 'blur', 'sideways']) { const rs = rows.filter((r) => r.shot === m); if (rs.length) console.log(m.padEnd(26) + String(rs.length).padStart(4) + `${rs.filter(ok).length} (${Math.round((100 * rs.filter(ok).length) / rs.length)}%)`.padStart(14)); }
}
if (hasTruth) {
  console.log('\n' + 'key fact'.padEnd(19) + 'pages'.padStart(6) + 'right'.padStart(12) + 'wrong'.padStart(7) + 'missing'.padStart(9) + '  shown as verified but wrong');
  for (const k of ['number', 'Full name', 'Date of birth', 'Expiry date']) {
    const rs = rows.filter((r) => r.facts?.[k]);
    const c = (v: string) => rs.filter((r) => r.facts![k] === v).length;
    if (rs.length) console.log(k.padEnd(19) + String(rs.length).padStart(6) + `${c('right')} (${Math.round((100 * c('right')) / rs.length)}%)`.padStart(12) + String(c('wrong')).padStart(7) + String(c('missing')).padStart(9) + String(c('verified-wrong')).padStart(8));
  }
  for (const t of types) { const rs = rows.filter((r) => r.label === t && r.facts); const all = rs.flatMap((r) => Object.values(r.facts!)); if (all.length) console.log(`  ${t.padEnd(17)} key facts right ${all.filter((v) => v === 'right').length}/${all.length} (${Math.round((100 * all.filter((v) => v === 'right').length) / all.length)}%)`); }
}
if (rows.length <= 60) {
  console.log('file'.padEnd(46) + 'sorted as'.padEnd(18) + 'sure'.padStart(5) + 'words'.padStart(7) + '  code lines');
  for (const r of rows) console.log((ok(r) ? '  ' : 'x ') + r.file.padEnd(44) + r.pred.padEnd(18) + `${Math.round(r.conf * 100)}%`.padStart(5) + String(r.words).padStart(7) + '  ' + r.mrz + (r.mrzFormat ? ' ' + r.mrzFormat : '') + (r.turned ? `  turned ${r.turned}` : ''));
}
fs.writeFileSync(path.join(DATA, `${root}-results-${split}.json`), JSON.stringify({ generated: new Date().toISOString(), rows }, null, 1));
