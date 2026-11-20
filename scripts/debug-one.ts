// Where does the time go on one image, and where was the page cut? Prints sizes, the page
// corners, timings and counts only. No text read from the image is ever printed.
// usage: npx tsx scripts/debug-one.ts <image> [folder to save the cut-out and cleaned images]
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { PNG } from 'pngjs';
import { createWorker, PSM } from 'tesseract.js';
import type { GrayImage } from '../src/core/types';
import { rgbaToGray, grayToRgba, preprocess, resize, ocrScale, backgroundUnevenness } from '../src/core/preprocess';
import { findDocument, flattenLighting, warpQuad } from '../src/core/scan';
import { wordsFromTesseract, cleanWords } from '../src/core/ocr';
import { readingQuality } from '../src/core/orient';

const file = process.argv[2], out = process.argv[3];
const ROOT = path.resolve(import.meta.dirname, '..');
const tess = await createWorker('eng', 1, { langPath: path.join(ROOT, 'node_modules/@tesseract.js-data/eng/4.0.0_best_int'), cacheMethod: 'none' });
await tess.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: '1' });
const png = (g: GrayImage) => { const p = new PNG({ width: g.width, height: g.height }); p.data = Buffer.from(grayToRgba(g)); return PNG.sync.write(p); };
const lap = (label: string, t: number) => console.log(label.padEnd(26), `${Date.now() - t} ms`);

let t = Date.now();
const meta = await sharp(file).rotate().metadata();
const k = Math.min(1, 2400 / Math.max(meta.width!, meta.height!));
const { data, info } = await sharp(file).rotate().flatten({ background: '#fff' }).resize(Math.round(meta.width! * k), undefined, { kernel: (process.env.KERNEL as any) ?? 'lanczos3' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
let color = new Uint8ClampedArray(data.buffer, data.byteOffset, data.length);
let W = info.width, H = info.height;
lap(`open ${W}x${H}`, t);
t = Date.now();
const g0 = rgbaToGray(color, W, H);
const quad = process.env.NOCUT ? null : findDocument(color, W, H, g0, process.env.FINDER !== 'bright');
lap('find page', t);
console.log('page corners (pixels):', quad ? quad.map((p) => p.map((v) => +v.toFixed(3)).join(',')).join('  ') : 'none (whole image kept)');
if (quad) { const w = warpQuad(color, W, H, quad); color = w.data as Uint8ClampedArray<ArrayBuffer>; W = w.width; H = w.height; console.log(`cut-out ${W}x${H}`); }
const uneven = backgroundUnevenness(quad ? rgbaToGray(color, W, H) : g0);
t = Date.now();
if (quad || uneven > 12) flattenLighting(color, W, H);
lap(`lighting (uneven ${uneven.toFixed(1)})`, t);
t = Date.now();
const scale = ocrScale(W, H);
const gray = preprocess(resize(rgbaToGray(color, W, H), scale), { deskew: true, binarise: false, removeLines: false }).img;
const clean = preprocess(gray, { deskew: false, binarise: true, removeLines: true }).img;
lap(`clean ${clean.width}x${clean.height}`, t);
let ink = 0; for (const v of clean.data) if (v < 128) ink++;
console.log('ink share of cleaned page:', (ink / clean.data.length).toFixed(3));
if (out) { fs.mkdirSync(out, { recursive: true }); fs.writeFileSync(path.join(out, 'gray.png'), png(gray)); fs.writeFileSync(path.join(out, 'clean.png'), png(clean)); }
for (const [name, img] of [['read cleaned page', clean], ['read grey page', gray]] as const) {
  t = Date.now();
  const { data: d } = await tess.recognize(png(img), {}, { blocks: true });
  const words = cleanWords(wordsFromTesseract(d.blocks as any, scale));
  lap(name, t);
  console.log(`  ${words.length} words, ${readingQuality(words)} clean words`);
}
await tess.terminate();
