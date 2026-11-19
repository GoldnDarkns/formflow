// Does "magic scan" (find page -> flatten -> fix lighting) help on phone photos?
// 20 FUNSD test forms were re-photographed synthetically on a cluttered,
// shadowed desk (see README). Compares OCR word recall with and without it.
// usage: npx tsx scripts/magic-eval.ts
import fs from 'node:fs'; import path from 'node:path';
import sharp from 'sharp';
import { createWorker, PSM } from 'tesseract.js';
import { rgbaToGray, grayToRgba, preprocess, resize, ocrScale } from '../src/core/preprocess';
import { detectPage, flattenLighting, warpQuad } from '../src/core/scan';
import { wordsFromTesseract, cleanWords } from '../src/core/ocr';
import { DATA, loadGt } from './funsd';

const ROOT = path.resolve(import.meta.dirname, '..');
const dir = path.join(DATA, 'photos');
const corners = JSON.parse(fs.readFileSync(path.join(dir, 'corners.json'), 'utf8'));
const tess = await createWorker('eng', 1, { langPath: path.join(ROOT, 'node_modules/@tesseract.js-data/eng/4.0.0_best_int'), cacheMethod: 'none' });
await tess.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: '1' });
const key = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');

async function ocr(rgba: Uint8ClampedArray, w: number, h: number) {
  let g = rgbaToGray(rgba, w, h);
  g = preprocess(resize(g, ocrScale(w, h))).img;
  const png = await sharp(Buffer.from(grayToRgba(g)), { raw: { width: g.width, height: g.height, channels: 4 } }).png().toBuffer();
  const { data } = await tess.recognize(png, {}, { blocks: true });
  return new Set(cleanWords(wordsFromTesseract(data.blocks as any)).map((w) => key(w.text)).filter(Boolean));
}
const tot = { gt: 0, plain: 0, cropOnly: 0, magic: 0, found: 0, n: 0, cornerErr: 0, ms: 0 };
for (const name of Object.keys(corners)) {
  const { data, info } = await sharp(path.join(dir, name.replace('.png', '.jpg'))).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const rgba = new Uint8ClampedArray(data);
  const gt = loadGt('testing', name.replace('.png', '.json')).ents.flatMap((e) => e.words).map((w) => key(w.text)).filter((t) => t.length >= 3);
  const plain = await ocr(rgba, info.width, info.height);
  const t = performance.now();
  const q = detectPage(rgbaToGray(rgba, info.width, info.height));
  let crop = { data: rgba, width: info.width, height: info.height };
  if (q) {
    tot.found++;
    const diag = Math.hypot(info.width, info.height);
    tot.cornerErr += q.reduce((a, p, i) => a + Math.hypot(p[0] - corners[name][i][0], p[1] - corners[name][i][1]), 0) / 4 / diag;
    crop = warpQuad(rgba, info.width, info.height, q) as any;
  }
  const cropOnly = await ocr(crop.data, crop.width, crop.height);
  const lit = new Uint8ClampedArray(crop.data);
  flattenLighting(lit, crop.width, crop.height);
  tot.ms += performance.now() - t;
  const magic = await ocr(lit, crop.width, crop.height);
  tot.n++; tot.gt += gt.length;
  tot.plain += gt.filter((w) => plain.has(w)).length; tot.cropOnly += gt.filter((w) => cropOnly.has(w)).length; tot.magic += gt.filter((w) => magic.has(w)).length;
  console.log(name, q ? 'page found' : 'NOT FOUND', (gt.filter((w) => plain.has(w)).length / gt.length).toFixed(2), '->', (gt.filter((w) => magic.has(w)).length / gt.length).toFixed(2));
}
const res = { photos: tot.n, pageFound: tot.found / tot.n, meanCornerError: tot.cornerErr / Math.max(1, tot.found), wordRecall: { photoAsTaken: tot.plain / tot.gt, pageCutOut: tot.cropOnly / tot.gt, cutOutAndLightingFixed: tot.magic / tot.gt } };
console.log(JSON.stringify(res, null, 1));
fs.writeFileSync(path.join(DATA, 'magic.json'), JSON.stringify(res));
await tess.terminate();
