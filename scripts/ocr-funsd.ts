// Run the classical pipeline (pre-processing + Tesseract) over FUNSD and cache
// the OCR words, so training and evaluation use exactly what the app produces.
// usage: npx tsx scripts/ocr-funsd.ts <split: training|testing> <mode: full|upscale|raw> [limit]

import fs from 'node:fs';
import path from 'node:path';
import { PNG } from 'pngjs';
import { createWorker, PSM } from 'tesseract.js';
import { rgbaToGray, grayToRgba, preprocess, resize, ocrScale } from '../src/core/preprocess';
import { wordsFromTesseract } from '../src/core/ocr';

const [split = 'testing', mode = 'full', limitArg] = process.argv.slice(2);
const root = path.resolve('../data/funsd/dataset', split + '_data');
const outDir = path.resolve('../data/ocr', `${split}-${mode}`);
fs.mkdirSync(outDir, { recursive: true });
const files = fs.readdirSync(path.join(root, 'images')).filter((f) => f.endsWith('.png')).sort();
const limit = limitArg ? +limitArg : files.length;

const langPath = path.resolve('node_modules/@tesseract.js-data/eng/4.0.0_best_int');

async function main() {
  const worker = await createWorker('eng', 1, { langPath, cacheMethod: 'none' });
  await worker.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: '1' });
  const t0 = Date.now();
  for (const f of files.slice(0, limit)) {
    const out = path.join(outDir, f.replace('.png', '.json'));
    if (fs.existsSync(out)) continue;
    const png = PNG.sync.read(fs.readFileSync(path.join(root, 'images', f)));
    let g = rgbaToGray(png.data, png.width, png.height);
    const scale = mode === 'raw' ? 1 : ocrScale(g.width, g.height);
    g = resize(g, scale);
    let skew = 0;
    if (mode === 'full') ({ img: g, skew } = preprocess(g));
    if (mode === 'deskew') ({ img: g, skew } = preprocess(g, { deskew: true, binarise: false, removeLines: false }));
    if (mode === 'nolines') ({ img: g, skew } = preprocess(g, { deskew: true, binarise: true, removeLines: false }));
    const enc = new PNG({ width: g.width, height: g.height });
    enc.data = Buffer.from(grayToRgba(g));
    const t = Date.now();
    const { data } = await worker.recognize(PNG.sync.write(enc), {}, { blocks: true });
    const words = wordsFromTesseract(data.blocks as any, scale);
    fs.writeFileSync(out, JSON.stringify({ file: f, width: png.width, height: png.height, skew, ms: Date.now() - t, words }));
    process.stdout.write(`${f} ${words.length}w ${Date.now() - t}ms\n`);
  }
  console.log('done', split, mode, ((Date.now() - t0) / 1000).toFixed(0) + 's');
  await worker.terminate();
}
main();
