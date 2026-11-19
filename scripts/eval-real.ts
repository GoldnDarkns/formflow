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
import { classifyDocument, DOC_TYPES } from '../src/core/classify';
import { believableMrz, mrzCrops, readingQuality, TURNS, TURN_GAIN, UPRIGHT_MIN } from '../src/core/orient';
import { DATA } from './funsd';

const split = process.argv[2] ?? 'train';
const ROOT = path.resolve(import.meta.dirname, '..');
const dir = path.join(DATA, 'own', split);
const cacheDir = path.join(DATA, 'own-cache', split);
const langPath = path.join(ROOT, 'node_modules/@tesseract.js-data/eng/4.0.0_best_int');
const MAX_SIDE = 2400;

const tess = await createWorker('eng', 1, { langPath, cacheMethod: 'none' });
await tess.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: '1' });
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

interface Row { file: string; label: string; pred: string; conf: number; words: number; pageFound: boolean; mrz: 'none' | 'found' | 'valid'; mrzFormat?: string; ms: number; turned: number }
const rows: Row[] = [];
const types = fs.existsSync(dir) ? fs.readdirSync(dir, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name) : [];
for (const label of types) {
  for (const f of fs.readdirSync(path.join(dir, label)).filter((x) => /\.(png|jpe?g|webp)$/i.test(x)).sort()) {
    const t0 = Date.now();
    const file = path.join(dir, label, f);
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
    for (const cr of mrzCrops(band, gray.height, words.length)) {
      const r = await tessMrz.recognize(png(cr.binarised ? clean : gray, cr.y0, cr.y1));
      const lines = findMrzLines(r.data.text);
      const m = lines.length >= 2 ? parseMrz(lines) : null;
      if (believableMrz(m, !!band) && mrzScore(m) > mrzScore(mrz)) mrz = m;
      if (mrz?.valid) break;
    }
    const cls = classifyDocument(words, W, H, mrz);
    fs.mkdirSync(path.join(cacheDir, label), { recursive: true });
    fs.writeFileSync(path.join(cacheDir, label, f.replace(/\.[^.]+$/, '.json')), JSON.stringify({ width: W, height: H, words, mrz: mrz ? { format: mrz.format, docType: mrz.docType, valid: mrz.valid, checks: mrz.checks.length, checksOk: mrz.checks.filter((c) => c.ok).length, country: mrz.fields.find((x) => x.key === 'Issuing country')?.value } : null }));
    rows.push({ file: f.replace(/\.[^.]+$/, ''), label, pred: cls.type, conf: cls.confidence, words: words.length, pageFound: !!quad, mrz: mrz ? (mrz.valid ? 'valid' : 'found') : 'none', mrzFormat: mrz?.format, ms: Date.now() - t0, turned });
    process.stdout.write('.');
  }
}
await tess.terminate();
await tessMrz.terminate();

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
const withMrz = rows.filter((r) => /pp-|eid-.*back|eid-.*together|eid-.*screenshot/.test(r.file));
console.log(`\ncode lines (passports and Emirates ID backs, ${withMrz.length} pages by file name): found ${withMrz.filter((r) => r.mrz !== 'none').length}, every check digit passes ${withMrz.filter((r) => r.mrz === 'valid').length}`);
console.log(`turned upright: ${rows.filter((r) => r.turned).length}`);
console.log(`page outline found in ${rows.filter((r) => r.pageFound).length} of ${rows.length}; median ${[...rows].sort((a, b) => a.ms - b.ms)[rows.length >> 1]?.ms} ms per page\n`);
console.log('file'.padEnd(46) + 'sorted as'.padEnd(18) + 'sure'.padStart(5) + 'words'.padStart(7) + '  code lines');
for (const r of rows) console.log((ok(r) ? '  ' : 'x ') + r.file.padEnd(44) + r.pred.padEnd(18) + `${Math.round(r.conf * 100)}%`.padStart(5) + String(r.words).padStart(7) + '  ' + r.mrz + (r.mrzFormat ? ' ' + r.mrzFormat : '') + (r.turned ? `  turned ${r.turned}` : ''));
fs.writeFileSync(path.join(DATA, `own-results-${split}.json`), JSON.stringify({ generated: new Date().toISOString(), rows }, null, 1));
